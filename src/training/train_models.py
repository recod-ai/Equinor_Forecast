# src/training/train_models.py
# src/training/train_models.py

# flake8: noqa: E402
"""Core model training and evaluation logic."""

# --- Standard Library Imports ---
from __future__ import annotations
import copy
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Third-Party Imports ---
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

# --- Local Application Imports ---
from evaluation.evaluation import history_evaluation
from models import create_model

# --- Optional Imports ---
# Gracefully import optional optimizers like LAMB
try:
    from tensorflow_addons.optimizers import LAMB
except ImportError:
    LAMB = None

# --- Module-level Configuration ---
# Set Matplotlib log level to warning to reduce startup verbosity
matplotlib.set_loglevel("warning")

# =============================================================================================================================================================
#                      --- Module Function Roadmap ---
# =============================================================================================================================================================
# | Function Name                          | Key Role                     | Purpose                                                                            |
# |----------------------------------------|------------------------------|------------------------------------------------------------------------------------|
# | `main_train_model`                     | **Primary Entry Point**      | Orchestrates model creation, configuration setup, and training dispatch.           |
# | `train_model`                          | **Training Dispatcher**      | Routes training execution to standard Keras, GradientTape, or staged modes.        |
# | `train_modern`                         | Standard Keras Loop          | Executes model training using `model.fit`, applying Cosine Decay and Snapshots.    |
# | `train_hybrid_staged` / `_three_stages`| Staged Hybrid Training       | Implements multi-stage training for hybrid models (e.g., Trend, Physics, Fusion).  |
# | `train_with_tape`                      | Custom Training Loop         | Implements a custom loop using TensorFlow's `GradientTape` for specialized losses. |
# | `get_optimizer`                        | Optimizer Factory            | Creates and configures optimizers (e.g., AdamW, Adam) with gradient clipping.      |
# | `SnapshotSaver`                        | Callback (Ensemble)          | Keras Callback that saves model weights at optimal points during LR cycles.        |
# | `prepare_training_sets_with_augmentation`| Data Prep Helper           | Augments or combines training/validation sets based on the chosen strategy.        |
# | `evaluate_snapshots_and_ensemble`      | Diagnostics                  | Assesses performance diversity of snapshot weights for ensemble creation.          |
# =============================================================================================================================================================


"""
Data Augmentation Impact and Shuffling in Time Series Regression:

1. Impact of Data Augmentation on Distribution:
   - Augmentation via scaling factors introduces synthetic data with varied scales, potentially skewing the dataset's 
     distribution (e.g., excessive low-amplitude examples from downscaled segments).
   - Without Shuffling: 
     - Batches may contain contiguous blocks of examples from a single scale regime (e.g., all small/large values), 
       causing model overfitting to local distributions. Validation loss may suffer if test data does not match the 
       skewed training regime (e.g., predictions biased toward low-amplitude values).
   - With Shuffling: 
     - Batches mix examples from all scales, promoting generalization across the full range of augmented and original data.

2. Why Shuffling Helps:
   - Balanced Batches: Ensures diverse scale representation per batch, mitigating overfitting to specific temporal segments.
   - Stabilized Gradients: Homogeneous batches can destabilize gradients (e.g., MAE loss sensitivity to target scale); 
     shuffling ensures gradient updates reflect the global data distribution.
   - Generalization: Test data retains original scales. Shuffling trains the model on all scales, reducing bias toward 
     augmented extremes (e.g., low values) and improving alignment with test conditions.

3. When Shuffling Is Acceptable:
   - Applicable if:
     - The task is regression (not direct forecasting) with time as an input feature, not a sequential dependency.
     - The model learns a time-agnostic input-output mapping.
   - Requirements:
     - No data leakage (e.g., strict train-test split preserving temporal integrity in test data).
     - Task goals prioritize scale-invariant predictions over temporal dynamics.
"""

# =============================================================================
# Helper Functions for Shared Configuration
# =============================================================================

def get_lr_schedule(
    initial_lr: float = 1e-3,
    first_decay_steps: int = 500
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Returns a cosine decay with restarts learning rate schedule."""
    return tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=1e-6
    )

# =============================================================================
# Função auxiliar para gerar um caminho único para checkpoint
# =============================================================================
def unique_checkpoint_path(checkpoint_path: str) -> str:
    """
    Generates a unique path for a checkpoint file by embedding the process ID.
    This prevents conflicts when running multiple training processes in parallel.
    """
    base, ext = os.path.splitext(checkpoint_path)
    unique_suffix = f"{os.getpid()}"
    return f"{base}_{unique_suffix}{ext}"

def get_callbacks(
    patience: int,
    checkpoint_path: str = 'best_model.keras',
    use_lr_scheduler: bool = False
) -> List[tf.keras.callbacks.Callback]:
    """
    Returns a list of callbacks including early stopping and model checkpointing.
    
    If use_lr_scheduler is True, a ReduceLROnPlateau callback is also included.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    if use_lr_scheduler:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.25,
                patience=50,
                min_lr=1e-6,
                verbose=1
            )
        )
        
    return callbacks

# =============================================================================
# Training Functions
# =============================================================================

def get_optimizer(
    optimizer_type: str = 'adamw',
    initial_lr: float = 1e-3,
    weight_decay: float = 1e-3,
    first_decay_steps: int = 500,
    global_clipnorm: float = 1.0
) -> tf.keras.optimizers.Optimizer:
    """
    Returns an optimizer instance based on the specified type.

    Args:
      optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd', 'rmsprop', 'adagrad').
      initial_lr: The initial learning rate.
      weight_decay: Weight decay value (only applied to AdamW).
      first_decay_steps: Steps for the first decay cycle (used in LR schedule if applicable).
      global_clipnorm: Global gradient clipping value.

    Returns:
      A configured instance of tf.keras.optimizers.Optimizer.
    """
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adamw':
        lr_schedule = get_lr_schedule(initial_lr, first_decay_steps)
        return tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            global_clipnorm=global_clipnorm
        )
    
    # For other optimizers, create a dictionary to reduce boilerplate
    optimizer_map = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'adagrad': tf.keras.optimizers.Adagrad,
    }

    if optimizer_type in optimizer_map:
        optimizer_class = optimizer_map[optimizer_type]
        config = {'learning_rate': initial_lr}
        # Add clipnorm/clipvalue based on optimizer support
        if optimizer_type == 'adam':
            config['clipvalue'] = global_clipnorm
        else:
            config['clipnorm'] = global_clipnorm
        
        # SGD has a momentum parameter that can be configured if needed
        if optimizer_type == 'sgd':
            config['momentum'] = 0.0

        return optimizer_class(**config)
    
    raise ValueError(f"Unsupported optimizer type: '{optimizer_type}'. "
                     "Choose from 'adamw', 'adam', 'sgd', 'rmsprop', or 'adagrad'.")



class HistoryObject:
    def __init__(self):
        self.history = {"loss": [], "val_loss": []}

def train_step(model, loss_fn, optimizer, X, y, diagnostic_interval, epoch, step):
    with tf.GradientTape(persistent=True) as tape:
        preds = model(X, training=True)
        # Separa a parte física dos resultados a partir do horizonte definido
        horizon = loss_fn.horizon
        physics_residual = preds[..., horizon:]
        # Calcula somente a loss física
        physics_loss = tf.reduce_mean(tf.abs(physics_residual))
        loss_value = physics_loss  # Usa apenas a loss física como loss final

    # Calcula os gradientes considerando as variáveis do modelo e do loss_fn
    grads = tape.gradient(loss_value, model.trainable_variables + loss_fn.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables + loss_fn.trainable_variables))

    iteration = epoch * diagnostic_interval + step
    if iteration % diagnostic_interval == 0:
        # Atualize a função diagnose_gradients para receber apenas a loss física, se necessário
        diagnose_gradients(tape, model, loss_fn, physics_loss, iteration)

    del tape
    return loss_value.numpy()



def diagnose_gradients(tape, model, loss_fn, trend_loss, physics_loss, iteration):
    trend_grads = tape.gradient(trend_loss, model.trainable_variables)
    physics_grads = tape.gradient(physics_loss, model.trainable_variables)
    alpha_grads = tape.gradient(loss_fn.log_alpha, loss_fn.trainable_variables)

    trend_mag = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in trend_grads if g is not None])
    physics_mag = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in physics_grads if g is not None])

    tf.print(f"\n[Diagnostics Iteration {iteration}]")
    tf.print("Trend Grad Magnitude:", trend_mag)
    tf.print("Physics Grad Magnitude:", physics_mag)
    tf.print("Alpha Gradient:", alpha_grads)
    tf.print("Current Alpha:", tf.sigmoid(loss_fn.log_alpha))

def create_tf_datasets(X, y, batch_size, val_split=0.1):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    # Embaralha o dataset com um buffer do tamanho do dataset
    dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    val_size = int(len(X) * val_split)
    # Separa os dados em validação e treinamento
    val_ds = dataset.take(val_size).batch(batch_size)
    train_ds = dataset.skip(val_size).batch(batch_size)
    return train_ds, val_ds


def evaluate_model(model, loss_fn, val_ds):
    val_losses = []
    for batch_X, batch_y in val_ds:
        preds = model(batch_X, training=False)
        loss = loss_fn(batch_y, preds)
        val_losses.append(loss.numpy())
    return np.mean(val_losses)


def train_with_tape(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    X: Union[np.ndarray, List[np.ndarray]],
    y: np.ndarray,
    epochs: int = 500,
    batch_size: int = 32,
    patience: int = 300,
    checkpoint_path: str = 'best_model.h5'
) -> Tuple[tf.keras.Model, dict]:
    """
    Trains a model using a custom GradientTape loop.

    This function mimics the behavior of `model.fit`, including dataset creation,
    evaluation, early stopping, and checkpointing.
    """
    train_ds, val_ds = create_tf_datasets(X, y, batch_size)
    best_val_loss = np.inf
    epochs_no_improve = 0
    history = {"loss": [], "val_loss": []}  # Use a standard dictionary
    
    @tf.function
    def train_step(batch_X, batch_y):
        with tf.GradientTape() as tape:
            preds = model(batch_X, training=True)
            loss = tf.keras.losses.MeanAbsoluteError()(batch_y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_losses = []
        for batch_X, batch_y in train_ds:
            loss = train_step(batch_X, batch_y)
            epoch_losses.append(loss.numpy())
        
        avg_epoch_loss = np.mean(epoch_losses)
        val_loss = evaluate_model(model, tf.keras.losses.MeanAbsoluteError(), val_ds)
        
        history["loss"].append(avg_epoch_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_weights(checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_weights(checkpoint_path)
    
    # To call history_evaluation, we wrap the dict in a temporary object
    class HistoryWrapper:
        def __init__(self, history_dict):
            self.history = history_dict
            
    history_evaluation(HistoryWrapper(history))
    
    return model, history


def plot_val_loss_snapshots(history, snapshot_epochs):
    val_loss = history["val_loss"]
    epochs = range(len(val_loss))
    plt.figure(figsize=(9,5))
    plt.plot(epochs, val_loss, label="Val Loss")
    for i, epoch in enumerate(snapshot_epochs):
        plt.axvline(epoch, color="r", linestyle="--", alpha=0.6, label="Snapshot" if i==0 else "")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.title("Val Loss & Snapshots")
    plt.show()

def compare_snapshots_weights(model, snapshot_weights):
    # Distância Euclidiana entre snapshots
    flattened = [np.concatenate([w.flatten() for w in weights]) for weights in snapshot_weights]
    print("--- Distância Euclidiana entre Snapshots ---")
    for i in range(len(flattened)):
        for j in range(i+1, len(flattened)):
            dist = np.linalg.norm(flattened[i] - flattened[j])
            print(f"Distância entre snapshot {i} e {j}: {dist:.2e}")

def evaluate_snapshots_and_ensemble(model, X_val, y_val, snapshot_weights, snapshot_epochs, history):
    # a) Plota curva val_loss + snapshots
    plot_val_loss_snapshots(history, snapshot_epochs)

    # b) Checa distância dos pesos
    compare_snapshots_weights(model, snapshot_weights)

    # c) Avaliação individual e ensemble
    individual_scores = []
    all_preds = []
    print("\n--- Avaliação de cada Snapshot ---")
    for idx, weights in enumerate(snapshot_weights):
        model.set_weights(weights)
        preds = model.predict(X_val, verbose=0)
        mse = np.mean((preds - y_val) ** 2)
        print(f"Snapshot {idx} (epoch {snapshot_epochs[idx]}): Val MSE = {mse:.4f}")
        all_preds.append(preds)
        individual_scores.append(mse)
    # Ensemble simples (média)
    ensemble_preds = np.mean(np.stack(all_preds), axis=0)
    ensemble_mse = np.mean((ensemble_preds - y_val) ** 2)
    print(f"\nEnsemble: Val MSE = {ensemble_mse:.4f}")
    return individual_scores, ensemble_mse



# =============================================================================
# WarmUpSchedule para facilitar o treino
# =============================================================================
class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup followed by any decay schedule.
    """
    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
        warmup_steps: int,
        name: str = None,
        dtype=tf.float32,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule = decay_schedule
        self.warmup_steps = warmup_steps
        self.name = name
        self.dtype = dtype

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        wm_f = tf.cast(self.warmup_steps, tf.float32)
        return tf.cond(
            step_f < wm_f,
            lambda: self.initial_learning_rate * (step_f / wm_f),
            lambda: self.decay_schedule(step - self.warmup_steps),
        )

    def get_config(self):
        cfg = self.decay_schedule.get_config()
        cfg.update({
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
            "dtype": self.dtype,
        })
        return cfg



def _get_lr_schedule(
    initial_lr: float,
    epochs: int,
    batch_size: int,
    num_samples: int,
    warmup_ratio: float = 0.1
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Build a warm-up + polynomial decay learning rate schedule.
    """
    steps_per_epoch = math.ceil(num_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    decay_steps = total_steps - warmup_steps

    poly_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=1e-5,
        power=1.0,
    )
    
    return WarmUpSchedule(initial_lr, poly_decay, warmup_steps)

def _get_lr_schedule_cosine_restarts(
    initial_lr: float,
    epochs: int,
    batch_size: int,
    num_samples: int,
    warmup_ratio: float = 0.1,
    cycles: int = 5, # O mesmo 'cycles' da SnapshotSaver
    t_mul: float = 2.0, # Multiplicador para a duração de cada ciclo
    m_mul: float = 0.5  # Multiplicador para o LR em cada restart
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    steps_per_epoch = math.ceil(num_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    
    first_decay_steps_for_cosine = (total_steps - warmup_steps) // cycles 

    cosine_decay_restarts = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps_for_cosine, # Duração do primeiro ciclo em passos
        t_mul=t_mul, # Cada ciclo subsequente é t_mul vezes mais longo
        m_mul=m_mul, # LR no restart é m_mul vezes o LR do ciclo anterior
        alpha=1e-5 # LR mínimo como fração do initial_lr
    )
    return WarmUpSchedule(initial_lr, cosine_decay_restarts, warmup_steps)


def _compile_model(
    model: tf.keras.Model,
    lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
    weight_decay: float,
    optimizer_type: str
) -> tf.keras.Model:
    """
    Compile the model with the selected optimizer and metrics.
    """
    if optimizer_type.lower() == 'lamb' and LAMB is not None:
        optimizer = LAMB(learning_rate=lr_schedule, weight_decay=weight_decay, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr_schedule, weight_decay=weight_decay, clipnorm=1.0, clipvalue=0.5
        )    

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanSquaredError()],
        steps_per_execution=1,
    )
    return model


"""
Training strategy rationale: epochs, batch size, and cosine decay cycles
-------------------------------------------------------------------------

This training setup is designed to balance convergence speed, model stability, and generalization using CosineDecayRestarts and Snapshot Ensembling. 

Key principles:
- Larger batch sizes reduce update frequency per epoch and require more epochs or fewer cycles.
- Cosine decay cycles must be long enough (≥10 epochs or ≥300 steps) to be meaningful.
- Snapshot ensembles work best with 3–5 cycles and sufficient learning rate decay range per cycle.
- Shuffle=True is recommended to avoid batch bias and promote better generalization.

Practical combined recommendations:

| Dataset Size     | Epochs | Batch Size | Cycles | Notes                                  |
|------------------|--------|------------|--------|----------------------------------------|
| Small (<10k)     | 100    | 32–64      | 3      | More steps, batches add regularization |
| Medium (~100k)   | 100    | 64–128     | 5      | Balanced cycles and convergence        |
| Large (1M+)      | 100–150| 128–256    | 3–5    | Long cycles, stable updates            |

Tips:
- Ensure steps_per_cycle ≥ 300 for stable cosine schedules.
- Use shuffle=True unless preserving sequence across samples is required.
- For Snapshot Ensembles, disable early stopping or set high patience.
"""


# =============================================================================
# Salva os Snapshots para usar com Ensembles + Armazena as epochs dos snapshots
# =============================================================================
class SnapshotSaver(tf.keras.callbacks.Callback):
    def __init__(self, epochs, cycles, steps_per_epoch):
        super().__init__()
        self.epochs = epochs
        self.cycles = cycles
        self.steps_per_epoch = steps_per_epoch
        self.epochs_per_cycle = math.ceil(epochs / cycles)
        self.best_loss_cycle = [np.inf] * cycles
        self.best_weights_cycle = [None] * cycles
        self.snapshot_epochs = [None] * cycles
        self.snapshot_val_losses = [None] * cycles  # <-- novo

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        cycle = epoch // self.epochs_per_cycle
        if cycle >= self.cycles:
            cycle = self.cycles - 1

        if val_loss < self.best_loss_cycle[cycle]:
            self.best_loss_cycle[cycle] = val_loss
            self.best_weights_cycle[cycle] = [w.copy() for w in self.model.get_weights()]
            self.snapshot_epochs[cycle] = epoch
            self.snapshot_val_losses[cycle] = val_loss


def _prepare_validation_data(
    X: Union[np.ndarray, List[np.ndarray]],
    y: np.ndarray,
    X_val: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    y_val: Optional[np.ndarray] = None,
    validation_split: float = 0.1,
    mode: str = "hybrid"
) -> tuple:
    """
    Prepare validation data according to the selected mode.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Training labels.
    X_val : array-like, optional
        Explicit validation features.
    y_val : array-like, optional
        Explicit validation labels.
    validation_split : float
        Proportion of X/y to split for validation (only used in 'classic' and 'hybrid' modes).
    mode : str
        How to assemble validation data. Options:
            - 'classic': use only a split of X/y for validation.
            - 'explicit': use only the explicit X_val/y_val provided.
            - 'hybrid': combine both (default): split X/y and concatenate with explicit X_val/y_val.

    Returns
    -------
    X_train, y_train, X_val_final, y_val_final : tuple
        Split training data and validation data as per the selected mode.

    Rationale
    ---------
    'hybrid' mode creates a more robust validation set by concatenating synthetically augmented data
    (coming from the split) with the external validation data. This provides better signal for hyperparameter
    selection, especially when synthetic data distributions differ from natural data.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    if mode not in ("classic", "explicit", "hybrid"):
        raise ValueError(f"Unknown validation mode: {mode}")

    if mode == "classic" or (X_val is None or y_val is None):
        # Only use a split of the training data
        X_train, X_val_split, y_train, y_val_split = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        return X_train, y_train, X_val_split, y_val_split

    if mode == "explicit":
        # Only use provided validation data; all X/y become training
        return X, y, X_val, y_val

    if mode == "hybrid":
        # Split a portion from training, then concatenate with provided validation data
        X_train, X_val_split, y_train, y_val_split = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        if isinstance(X_val_split, list):
            # For multi-input (list of arrays)
            X_val_final = [np.concatenate([xv, xve], axis=0) for xv, xve in zip(X_val_split, X_val)]
        else:
            X_val_final = np.concatenate([X_val_split, X_val], axis=0)
        y_val_final = np.concatenate([y_val_split, y_val], axis=0)
        return X_train, y_train, X_val_final, y_val_final

def select_best_snapshots_from_callback(
    snapshot_callback,
    earlystopping_callback=None,
    epochs=None,
    n_best=3,
):
    """
    Retorna listas dos N melhores snapshots (pesos, epochs, val_losses) a partir do callback.
    Se não houver snapshots, usa os pesos do EarlyStopping.
    """
    # Extrai snapshots válidos
    snapshot_weights = [w for w in getattr(snapshot_callback, "best_weights_cycle", []) if w is not None]
    snapshot_epochs  = [e for e in getattr(snapshot_callback, "snapshot_epochs", []) if e is not None]
    snapshot_val_losses = [l for l in getattr(snapshot_callback, "snapshot_val_losses", []) if l is not None]

    # Caso nenhum snapshot válido (fallback EarlyStopping)
    if not snapshot_weights and earlystopping_callback is not None:
        print("Warning: No snapshot cycles completed. Using best weights from EarlyStopping.")
        snapshot_weights = [getattr(earlystopping_callback, "best_weights", None)]
        snapshot_epochs = [epochs-1] if epochs is not None else [None]
        snapshot_val_losses = [np.nan]

    # Seleciona os N melhores snapshots pelo menor val_loss
    if len(snapshot_weights) > n_best and len(snapshot_val_losses) == len(snapshot_weights):
        idx_best = np.argsort(snapshot_val_losses)[:n_best]
        snapshot_weights = [snapshot_weights[i] for i in idx_best]
        snapshot_epochs = [snapshot_epochs[i] for i in idx_best]
        snapshot_val_losses = [snapshot_val_losses[i] for i in idx_best]
    
    return snapshot_weights, snapshot_epochs, snapshot_val_losses



import numpy as np
import statsmodels.api as sm
from typing import Tuple, Union, List

# Place this new helper function somewhere accessible.
def create_hp_filtered_data(
    X: np.ndarray, 
    y: np.ndarray, 
    hp_lambda: float = 128000.0,
    features_to_filter: List[int] = [0, 3, 7] # PI, AVG_DOWNHOLE_PRESSURE, BORE_OIL_VOL
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a smoothed, synthetic version of windowed (X, y) data using an HP filter.

    Args:
        X: Input features of shape (n_samples, timesteps, n_features).
        y: Target values of shape (n_samples, horizon).
        hp_lambda: The smoothing parameter for the HP filter.
        features_to_filter: List of indices of the features in X to smooth.

    Returns:
        A tuple of (X_filtered, y_filtered).
    """
    print(f"INFO: Creating HP-filtered synthetic data. Smoothing features at indices: {features_to_filter}")
    X_filtered = np.copy(X)
    y_filtered = np.copy(y)

    for i in range(X.shape[0]):
        # Filter specified features in X for each sample
        for feat_idx in features_to_filter:
            if feat_idx < X.shape[2]: # Safety check
                try:
                    # HP filter returns (cycle, trend), we want the smooth trend
                    _, trend = sm.tsa.filters.hpfilter(X[i, :, feat_idx], lamb=hp_lambda)
                    X_filtered[i, :, feat_idx] = trend
                except Exception as e:
                    print(f"Warning: HP filter failed for sample {i}, feature {feat_idx}. Keeping original. Error: {e}")

        # Filter the target y for each sample
        try:
            _, trend = sm.tsa.filters.hpfilter(y[i, :], lamb=hp_lambda)
            y_filtered[i, :] = trend
        except Exception as e:
            print(f"Warning: HP filter failed for target sample {i}. Keeping original. Error: {e}")
            
    return X_filtered, y_filtered

# Place this function right above train_modern.
def prepare_training_sets_with_augmentation(
    X_train_orig: np.ndarray,
    y_train_orig: np.ndarray,
    X_val_orig: np.ndarray,
    y_val_orig: np.ndarray,
    mode: str = "classic", # "classic" | "augment_train" | "fold_in_val"
    hp_lambda: float = 512000.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Orchestrates data preparation, providing different augmentation strategies.

    Args:
        X_train_orig, y_train_orig: The original training data.
        X_val_orig, y_val_orig: The original validation data.
        mode: The strategy to use.
            - "classic": Use original data as is.
            - "augment_train": Augment training data with a filtered version of itself.
            - "fold_in_val": Augment training data with a filtered version of the validation set.

    Returns:
        A tuple of the final (X_train, y_train, X_val, y_val) to be used in model.fit().
    """
    if mode == "classic":
        logging.info("INFO: Data prep mode 'classic'. Using original data.")
        return X_train_orig, y_train_orig, X_val_orig, y_val_orig

    elif mode == "augment_train":
        logging.info("INFO: Data prep mode 'augment_train'. Filtering train set and adding to itself.")
        X_filtered, y_filtered = create_hp_filtered_data(X_train_orig, y_train_orig, hp_lambda)
        
        X_train_final = np.concatenate([X_train_orig, X_filtered], axis=0)
        y_train_final = np.concatenate([y_train_orig, y_filtered], axis=0)
        
        # Validation set remains untouched
        return X_train_final, y_train_final, X_val_orig, y_val_orig

    elif mode == "fold_in_val":
        logging.info("INFO: Data prep mode 'fold_in_val'. Adding to train.")
        X_train_final = np.concatenate([X_train_orig, X_val_orig], axis=0)
        y_train_final = np.concatenate([y_train_orig, y_val_orig], axis=0)

        # Validation set remains untouched
        return X_train_final, y_train_final, X_val_orig, y_val_orig
        
    else:
        raise ValueError(f"Unknown data preparation mode: '{mode}'")


# ========= Base =========
class BaseTransform:
    """Contrato: recebe (X, y) e devolve (X', y') SEM mutar inputs."""
    def __init__(self, **kwargs):
        self.cfg = dict(kwargs or {})

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.apply(X, y)

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class HPFilterTransform(BaseTransform):
    """
    Suaviza features/target com HP-filter.
    modes:
      - 'two_sided'     : HP tradicional (NÃO causal). Use só no treino.
      - 'composite_base': seu método "smoothed-history base" por janelas (causal por passo).
    """
    def __init__(
        self,
        hp_lambda: float = 128000.0,
        mode: str = "two_sided",   # 'two_sided' | 'composite_base'
        features_to_filter: Optional[List[int]] = None,  # índices no eixo de features de X
        filter_target: bool = True,
    ):
        super().__init__(
            hp_lambda=hp_lambda, mode=mode,
            features_to_filter=features_to_filter, filter_target=filter_target
        )
        self.lamb = float(hp_lambda)
        self.mode = mode
        self.features_to_filter = features_to_filter or []
        self.filter_target = bool(filter_target)

    def _hp_series(self, series_1d: np.ndarray) -> np.ndarray:
        """HP two-sided numa série 1D; devolve o trend (suavizado)."""
        _, trend = sm.tsa.filters.hpfilter(series_1d, lamb=self.lamb)
        return np.asarray(trend, dtype=float)

    def _hp_composite_hist_base(self, windows_2d: np.ndarray) -> np.ndarray:
        """
        Sua ideia causal por passo: usa a fita 0 inteira como base suavizada,
        e depois agrega apenas o último ponto suavizado de cada janela subsequente.
        windows_2d: (N, H)
        return: série 1D de comprimento L = N + H - 1
        """
        n_windows, horizon = windows_2d.shape
        L = n_windows + horizon - 1
        out = np.empty(L, dtype=float)

        # 1) suaviza a 1ª fita inteira
        base_sm = self._hp_series(windows_2d[0, :])
        out[:horizon] = base_sm

        # 2) para cada janela i>0, pega só o último ponto (raw), faz HP sobre histórico + esse ponto, guarda último do trend
        for i in range(1, n_windows):
            t = horizon + i - 1
            next_raw_point = windows_2d[i, -1]  # causal: só usa o último da janela i
            composite = np.append(out[:t], next_raw_point)
            _, smoothed = sm.tsa.filters.hpfilter(composite, lamb=self.lamb)
            out[t] = float(smoothed[-1])
        return out

    def _series_to_windows(self, series: np.ndarray, N: int, H: int) -> np.ndarray:
        """Reconstrói janelas stride=1 a partir de série 1D (L=N+H-1)."""
        L = series.shape[0]
        exp_L = N + H - 1
        if L != exp_L:
            raise ValueError(f"series len {L} != N+H-1 ({exp_L})")
        out = np.empty((N, H), dtype=series.dtype)
        for i in range(N):
            out[i, :] = series[i:i+H]
        return out

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xf = np.array(X, copy=True)
        yf = np.array(y, copy=True)

        # --- Features (X: N, M, F) -> suaviza ao longo do tempo M ---
        if Xf.ndim != 3:
            raise ValueError("HPFilterTransform espera X com shape (N, M, F)")
        N, M, F = Xf.shape
        for fi in self.features_to_filter:
            if fi < 0 or fi >= F:
                continue
            if self.mode == "two_sided":
                for i in range(N):
                    Xf[i, :, fi] = self._hp_series(Xf[i, :, fi])
            elif self.mode == "composite_base":
                # composite_base faz sentido para *alvo* windowed; para X (histórico), usamos two_sided por amostra
                # para manter causal em X: podemos usar EMAFilterTransform em vez de HP composite.
                for i in range(N):
                    Xf[i, :, fi] = self._hp_series(Xf[i, :, fi])
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        # --- Target (y: N, H) ---
        if self.filter_target:
            if yf.ndim != 2:
                raise ValueError("HPFilterTransform espera y com shape (N, H)")
            Nw, H = yf.shape
            if self.mode == "two_sided":
                for i in range(Nw):
                    yf[i, :] = self._hp_series(yf[i, :])
            elif self.mode == "composite_base":
                # 1) gera série 1D suavizada causal
                series = self._hp_composite_hist_base(yf)
                # 2) volta para janelas stride=1
                yf = self._series_to_windows(series, Nw, H)
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        return Xf, yf


class StepSmootherTransform(BaseTransform):
    """
    Detecta 'degraus' por limiar em |Δ| e substitui pontos por média local.
    - causal=True -> usa média de janela [max(0,t-w), t]
    - causal=False -> usa média centrada [t-w, t+w]
    Aplica em X(features selecionadas) e/ou y.
    """
    def __init__(
        self,
        delta_threshold: float = 3_000.0,
        window: int = 3,
        causal: bool = True,
        features_to_filter: Optional[List[int]] = None,
        filter_target: bool = False,
    ):
        super().__init__(delta_threshold=delta_threshold, window=window, causal=causal,
                         features_to_filter=features_to_filter, filter_target=filter_target)
        self.th = float(delta_threshold)
        self.win = int(window)
        self.causal = bool(causal)
        self.features_to_filter = features_to_filter or []
        self.filter_target = bool(filter_target)

    def _smooth_steps(self, s: np.ndarray) -> np.ndarray:
        s = np.array(s, dtype=float, copy=True)
        dif = np.abs(np.diff(s, prepend=s[0]))
        idx = np.where(dif > self.th)[0]
        for t in idx:
            if self.causal:
                a = max(0, t - self.win)
                b = t + 1
            else:
                a = max(0, t - self.win)
                b = min(len(s), t + self.win + 1)
            m = np.mean(s[a:b])
            s[t] = m
        return s

    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xf = np.array(X, copy=True)
        yf = np.array(y, copy=True)

        if Xf.ndim != 3:
            raise ValueError("StepSmootherTransform espera X com shape (N, M, F)")
        N, M, F = Xf.shape
        for fi in self.features_to_filter:
            if 0 <= fi < F:
                for i in range(N):
                    Xf[i, :, fi] = self._smooth_steps(Xf[i, :, fi])

        if self.filter_target:
            if yf.ndim != 2:
                raise ValueError("StepSmootherTransform espera y com shape (N, H)")
            Nw, H = yf.shape
            for i in range(Nw):
                yf[i, :] = self._smooth_steps(yf[i, :])

        return Xf, yf


def apply_pipeline(
    X: np.ndarray, y: np.ndarray, transforms: List[BaseTransform]
) -> Tuple[np.ndarray, np.ndarray]:
    Xo, yo = X, y
    for t in transforms or []:
        Xo, yo = t(Xo, yo)
    return Xo, yo

def build_transforms_from_cfg(cfg: Optional[Dict[str, Any]]) -> List[BaseTransform]:
    """
    Exemplo de cfg:
    {
      "hp": {"enabled": True, "mode": "composite_base", "hp_lambda": 128000.0,
             "features_to_filter": [0,3], "filter_target": True},
      "step": {"enabled": True, "delta_threshold": 2500.0, "window": 2, "causal": True,
               "features_to_filter": [0,3], "filter_target": True},
    }
    """
    transforms: List[BaseTransform] = []
    if not cfg:
        return transforms

    hp_cfg = cfg.get("hp", {})
    if hp_cfg.get("enabled", False):
        transforms.append(
            HPFilterTransform(
                hp_lambda = hp_cfg.get("hp_lambda", 128000.0),
                mode = hp_cfg.get("mode", "two_sided"),
                features_to_filter = hp_cfg.get("features_to_filter", []),
                filter_target = hp_cfg.get("filter_target", True),
            )
        )

    ema_cfg = cfg.get("ema", {})
    if ema_cfg.get("enabled", False):
        transforms.append(
            EMAFilterTransform(
                alpha = ema_cfg.get("alpha", 0.2),
                features_to_filter = ema_cfg.get("features_to_filter", []),
                filter_target = ema_cfg.get("filter_target", False),
            )
        )

    step_cfg = cfg.get("step", {})
    if step_cfg.get("enabled", False):
        transforms.append(
            StepSmootherTransform(
                delta_threshold = step_cfg.get("delta_threshold", 3000.0),
                window = step_cfg.get("window", 3),
                causal = step_cfg.get("causal", True),
                features_to_filter = step_cfg.get("features_to_filter", []),
                filter_target = step_cfg.get("filter_target", False),
            )
        )

    return transforms



transform_cfg = {
    "hp": {
        "enabled": False,
        "mode": "composite_base",       # causal por passo (boa pra fronteira)
        "hp_lambda": 64000.0,
        "features_to_filter": [],       # não filtra X com HP aqui
        "filter_target": True,
    },

    # "step": {
    #     "enabled": True,
    #     "delta_threshold": 2500.0,
    #     "window": 2,
    #     "causal": True,
    #     "features_to_filter": [0, 3],   # remove degraus fortes nos drivers
    #     "filter_target": False,         # deixe False se já usar HP no y
    # },
}





def train_modern(
    model: tf.keras.Model,
    X: Union[np.ndarray, List[np.ndarray]],
    y: np.ndarray,
    X_val: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    y_val: Optional[np.ndarray] = None, 
    epochs: int = 300,
    batch_size: int = 32,
    patience: int = 100,
    optimizer_type: str = "adam",
    initial_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_path: str = 'best_model.keras',
    validation_split: float = 0.1,
    cycles: int = 5,
    use_mixed_precision: bool = True,
    validation_mode: str = "classic" #hybrid, explicit, | "augment_train" | "fold_in_val"
) -> Tuple[tf.keras.Model, dict, tuple]:
    """
    Train the model with a modern signature and flexible validation set creation.

    Parameters
    ----------
    ...
    validation_mode : str
        One of 'classic', 'explicit', or 'hybrid' (default). Controls how the validation set is formed.
        See `_prepare_validation_data` for details.
    """

    # Build learning rate COSINE
    lr_schedule = _get_lr_schedule_cosine_restarts(
        initial_lr=initial_lr,
        epochs=epochs,
        batch_size=batch_size,
        num_samples=len(y),
        warmup_ratio=0.1,
        cycles=cycles
    )

    # Compile the model
    model = _compile_model(
        model=model,
        lr_schedule=lr_schedule,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
    )

    steps_per_epoch = math.ceil(len(y) / batch_size)

    # Prepare callbacks
    snapshot_callback = SnapshotSaver(epochs=epochs, cycles=cycles, steps_per_epoch=steps_per_epoch)
    callbacks = [
        snapshot_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            restore_best_weights=True
        ),
        # OptunaPruningCallback(None, monitor='val_loss'),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
    ]

    X_train, y_train, X_val_final, y_val_final = prepare_training_sets_with_augmentation(
        X_train_orig=X,
        y_train_orig=y,
        X_val_orig=X_val,
        y_val_orig=y_val,
        mode=validation_mode
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val_final, y_val_final),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )

    snapshot_weights, snapshot_epochs, _ = select_best_snapshots_from_callback(
        snapshot_callback, 
        earlystopping_callback=callbacks[1], 
        epochs=epochs, 
        n_best=5
    )

    model._snapshot_weights = snapshot_weights
    # history_evaluation(history)
    # evaluate_snapshots_and_ensemble(model, X_val_final, y_val_final, snapshot_weights, snapshot_epochs, history.history)
    return model, history.history




def _train_individual_block(
    X_train: Union[np.ndarray, List[np.ndarray]],
    y_train: np.ndarray,
    X_val: Optional[Union[np.ndarray, List[np.ndarray]]],
    y_val: Optional[np.ndarray], 
    model_args: dict,
    freeze_trend: bool,
    freeze_physics: bool,
    fusion_type: str,
    epochs: int,
    batch_size: int,
    patience: int,
    optimizer_type: str,
    initial_lr: float,
    weight_decay: float,
    checkpoint_path: str,
) -> tf.keras.Model:
    """
    Train a single block of the hybrid model (Trend or Physics).
    """
    model = create_model(
        **model_args,
        freeze_trend=freeze_trend,
        freeze_physics=freeze_physics,
        fusion_type=fusion_type,
    )
    model, _ = train_modern(
        model=model,
        X=X_train,
        y=y_train,
        X_val=X_val, 
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=optimizer_type,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path
    )
    return model


def _assemble_fusion_model(
    model_args: dict,
    trend_model: tf.keras.Model,
    physics_model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Create a fusion model and load pretrained block weights.
    """
    fusion_model = create_model(**model_args, fusion_type="concat_dense")
    fusion_model.get_layer('trend_block').set_weights(
        trend_model.get_layer('trend_block').get_weights()
    )
    fusion_model.get_layer('physics_block').set_weights(
        physics_model.get_layer('physics_block').get_weights()
    )
    return fusion_model


def train_hybrid_staged(
    X_train: Union[np.ndarray, List[np.ndarray]],
    y_train: np.ndarray,
    X_val: Optional[Union[np.ndarray, List[np.ndarray]]],
    y_val: Optional[np.ndarray], 
    model_args: dict,
    epochs: int,
    batch_size: int,
    patience: int,
    optimizer_type: str,
    initial_lr: float,
    weight_decay: float,
    checkpoint_path: str,
    scaler_X: Any,
    scaler_target: Any
) -> Tuple[tf.keras.Model, dict]:
    """
    Stage-wise training for hybrid models combining Trend and Physics blocks:
      1. Train Trend block alone.
      2. Train Physics block alone.
      3. Train fusion layer with blocks frozen.
      4. Fine-tune complete model with all layers trainable.
    """
    # Stage 1: Trend block
    logging.info(">>> STAGE 1: Training TREND Block")
    trend_model = _train_individual_block(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val, 
        y_val=y_val, 
        model_args=model_args,
        freeze_trend=False,
        freeze_physics=True,
        fusion_type='trend',
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=optimizer_type,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        checkpoint_path='trend_only.keras',
    )

    # Stage 2: Physics block
    logging.info(">>> STAGE 2: Training PINN Block")
    physics_model = _train_individual_block(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val, 
        y_val=y_val,
        model_args=model_args,
        freeze_trend=True,
        freeze_physics=False,
        fusion_type='pin',
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=optimizer_type,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        checkpoint_path='pin_only.keras',
    )

    # Stage 3: Fusion layer training
    logging.info(">>> STAGE 3: Training FUSION Block")
    fusion_args = {**model_args, 'freeze_trend': True, 'freeze_physics': True}
    fusion_model = _assemble_fusion_model(fusion_args, trend_model, physics_model)
    fusion_model, history = train_modern(
        model=fusion_model,
        X=X_train,
        y=y_train,
        X_val=X_val, 
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=optimizer_type,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        checkpoint_path='fusion_only.keras'
    )

    # Stage 4: Fine-tuning all layers
    # logging.info(">>> STAGE 4: Fine-tuning all layers")
    # fusion_model.get_layer('trend_block').trainable = True
    # fusion_model.get_layer('physics_block').trainable = True
    # fusion_model, history = train_modern(
    #     model=fusion_model,
    #     X=X_train,
    #     y=y_train,
    #     X_val=X_val, 
    #     y_val=y_val,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     patience=patience,
    #     optimizer_type=optimizer_type,
    #     initial_lr=initial_lr,
    #     weight_decay=weight_decay,
    #     checkpoint_path=checkpoint_path
    # )

    return fusion_model, None

def train_hybrid_three_stages(
    X, y,
    X_val: Optional[Union[np.ndarray, List[np.ndarray]]],
    y_val: Optional[np.ndarray], 
    build_kwargs: dict,
    epochs_per_stage: int,
    batch_size: int,
    patience: int,
    optimizer_cfg: dict,
    initial_lr:float,
    val_split: float = 0.1,
    cycles: int = 5,
    use_mixed_precision: bool = True,
    scaler_X: Any = None,
    scaler_target: Any = None
) -> Tuple[tf.keras.Model, dict]:
    # ——————————————————————————————————————————
    # 1) monta o modelo “dict” com cabeça de extrator
    # ——————————————————————————————————————————
    model = create_model(
        **build_kwargs,
        output_mode='dict',
        add_extractor_head=True,
        scaler_X=scaler_X,
        scaler_target=scaler_target
    )

    # garante pasta de checkpoints
    ckpt_dir = "checkpoint"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt1 = f"{ckpt_dir}/s1_physics_best.keras"
    ckpt2 = f"{ckpt_dir}/s2_extractor_best.keras"
    ckpt3 = f"{ckpt_dir}/s3_fuser_combiner_best.keras"

    # extrai parâmetros do optimizador para o train_modern
    opt_cls     = optimizer_cfg
    wd0         = 1e-4

    histories = {}

    # ——————————————————————————————————————————
    # STAGE 1: Physics Block
    # ——————————————————————————————————————————
    logging.info(">>> STAGE 1: Treinando Physics Block")
    model.physics_block.trainable   = True
    model.extractor_block.trainable = False
    model.fuser_block.trainable     = False
    model.combiner_block.trainable  = False
    if hasattr(model, 'extractor_head_dense_block'):
        model.extractor_head_dense_block.trainable = False

    # wrapper que expõe só a saída physics_scaled
    stage1 = Model(
        inputs=model.input,
        outputs=model.get_layer('physics_scaled_output').output
    )
    _, h1 = train_modern(
        model=stage1,
        X=X, y=y,
        X_val=X_val, 
        y_val=y_val, 
        epochs=epochs_per_stage,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=opt_cls,
        initial_lr=initial_lr,
        weight_decay=wd0,
        checkpoint_path=ckpt1,
        validation_split=val_split,
        cycles=cycles,
        use_mixed_precision=use_mixed_precision
    )
    histories['stage1_physics'] = h1

    # ——————————————————————————————————————————
    # STAGE 2: Extractor + cabeça
    # ——————————————————————————————————————————
    logging.info(">>> STAGE 2: Treinando Extractor + Head")
    model.physics_block.trainable   = False
    model.extractor_block.trainable = True
    model.fuser_block.trainable     = False
    model.combiner_block.trainable  = False
    model.extractor_head_dense_block.trainable = True

    stage2 = Model(
        inputs=model.input,
        outputs=model.get_layer('extractor_head_dense_block').output
    )
    _, h2 = train_modern(
        model=stage2,
        X=X, y=y,
        X_val=X_val, 
        y_val=y_val, 
        epochs=epochs_per_stage,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=opt_cls,
        initial_lr=initial_lr,
        weight_decay=wd0,
        checkpoint_path=ckpt2,
        validation_split=val_split,
        cycles=cycles,
        use_mixed_precision=use_mixed_precision
    )
    histories['stage2_extractor'] = h2

    # ——————————————————————————————————————————
    # STAGE 3: Fuser + Combiner
    # ——————————————————————————————————————————
    logging.info(">>> STAGE 3: Treinando Fuser & Combiner")
    model.physics_block.trainable   = False
    model.extractor_block.trainable = False
    model.extractor_head_dense_block.trainable = False
    model.fuser_block.trainable     = True
    model.combiner_block.trainable  = True

    stage3 = Model(
        inputs=model.input,
        outputs=model.get_layer('final_forecast_output').output
    )
    _, h3 = train_modern(
        model=stage3,
        X=X, y=y,
        X_val=X_val, 
        y_val=y_val,
        epochs=epochs_per_stage,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=opt_cls,
        initial_lr=initial_lr,
        weight_decay=wd0,
        checkpoint_path=ckpt3,
        validation_split=val_split,
        cycles=cycles,
        use_mixed_precision=use_mixed_precision
    )
    histories['stage3_fuser_combiner'] = h3

    # ——————————————————————————————————————————
    # Modelo final de inferência (saída única)
    # ——————————————————————————————————————————
    stage3._name = "HybridInference"    # opcional, só pra renomear
    inference_model = stage3            # reaproveita o objeto que TEM snapshots


    return inference_model, histories

def train_model(
    model: tf.keras.Model,
    X_train: Union[np.ndarray, List[np.ndarray]],
    y_train: np.ndarray,
    X_val: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    y_val: Optional[np.ndarray] = None,                         
    epochs: int = 500,
    batch_size: int = 32,
    patience: int = 300,
    optimizer_type: str = 'adam',
    initial_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    first_decay_steps: int = 100,
    checkpoint_path: str = 'best_model.keras',
    use_gradient_tape: bool = False,
    training_mode: str = "traditional",
    architecture_name: Optional[str] = None,
    strategy_config: Optional[Dict] = None,
    extractor_config: Optional[Dict] = None,
    fuser_config: Optional[Dict] = None,
    scaler_X: Any = None,
    scaler_target: Any = None,
) -> Tuple[tf.keras.Model, dict]:
    """
    Train the model using traditional, GradientTape, or hybrid staged strategies.
    """
    optimizer = get_optimizer(optimizer_type, initial_lr, weight_decay, first_decay_steps)

    logging.info(f"TRAINING MODE: {training_mode}")
    logging.info(f"ARCHITECTURE KIND: {architecture_name}")

    if use_gradient_tape:
        return train_with_tape(
            model, optimizer, X_train, y_train, epochs, batch_size, patience, checkpoint_path
        )

    model_args = {
        "input_shape": X_train.shape[1:], # Use the shape tuple here too for consistency
        "horizon": y_train.shape[1],
        "strategy_config": strategy_config, 
        "extractor_config": extractor_config, 
        "fuser_config": fuser_config, 
        "architecture_name": architecture_name,
        "scaler_X": scaler_X,
        "scaler_target": scaler_target
    }

    
    if architecture_name == "Seq2Trend":
        return train_hybrid_staged(
            X_train,
            y_train,
            X_val,
            y_val,   
            model_args,
            epochs,
            batch_size,
            patience,
            optimizer_type,
            initial_lr,
            weight_decay,
            checkpoint_path,
            scaler_X=scaler_X,
            scaler_target=scaler_target            
        )

    if architecture_name == "Seq2Fuser":

        model_args = {
            "input_shape": X_train,
            "horizon": y_train.shape[1],
            "strategy_config": model.strategy_config,
            "architecture_name": architecture_name,
            "fuser_config": model.fuser_config,
            "extractor_config": model.extractor_config,
            "scaler_X": scaler_X,
            "scaler_target": scaler_target
        }

        return train_hybrid_three_stages(
            X_train, y_train,
            X_val,
            y_val, 
            build_kwargs = model_args,
            epochs_per_stage = epochs,
            batch_size = batch_size,
            patience = patience,
            optimizer_cfg = optimizer_type,
            initial_lr=initial_lr
        )

    return train_modern(
        model=model,
        X=X_train,
        y=y_train,
        X_val=X_val, 
        y_val=y_val,   
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=optimizer_type,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path
    )

def main_train_model(
    architecture_name: str,
    feature_kind: str,
    train_kwargs: Dict,
    prediction_input: Union[np.ndarray, List[np.ndarray]],
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 25,
    learning_rate: float = 1e-3,
    training_mode: str = "traditional"
) -> Tuple[tf.keras.Model, Dict, np.ndarray]:

    
    # Extract configs and data
    strategy_config = train_kwargs.get('strategy_config')
    extractor_config = train_kwargs.get('extractor_config')
    fuser_config = train_kwargs.get('fuser_config')

    dataset_name = train_kwargs.get('dataset_name')
    
    if strategy_config and dataset_name:
        strategy_config['dataset_name'] = dataset_name

    # Unpack the scaler objects from the train_kwargs dictionary.
    scaler_X = train_kwargs.get('scaler_X')
    scaler_target = train_kwargs.get('scaler_target')
    if scaler_X is None or scaler_target is None:
        raise ValueError("Scalers were not found in train_kwargs. They must be added during data preparation.")
    
    X_train = train_kwargs['X_train']
    y_train = train_kwargs['y_train']
    X_val   = train_kwargs.get('X_val')
    y_val   = train_kwargs.get('y_val')
    
    # Define shape as a tuple
    input_shape_tuple = X_train.shape[1:]

    # Create model creation args
    model_creation_args = {
        "input_shape": input_shape_tuple,
        "architecture_name": architecture_name,
        "scaler_X": scaler_X,
        "scaler_target": scaler_target,
    }
    
    SEQ2SEQ_ARCHS = ["Seq2Context", "Seq2PIN", "Seq2Trend", "Seq2Fuser"]
    if architecture_name in SEQ2SEQ_ARCHS:
        model_creation_args.update({
            "horizon": y_train.shape[1],
            "strategy_config": strategy_config,
            "extractor_config": extractor_config,
            "fuser_config": fuser_config,
        })

    model = create_model(**model_creation_args)

    return train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,          
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        initial_lr=learning_rate,
        training_mode=training_mode,
        architecture_name=architecture_name,
        strategy_config=strategy_config,
        extractor_config=extractor_config,
        fuser_config=fuser_config,
        scaler_X=scaler_X,
        scaler_target=scaler_target,
    )