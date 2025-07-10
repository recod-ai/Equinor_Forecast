# src/training/train_models.py
import os
import time
import math
import  logging
import numpy as np
from typing import Any, Dict, Tuple, Union, List, Optional
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from models import create_model
import tensorflow_addons as tfa  # For AdamW optimizer

import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Tuple, Union

from evaluation.evaluation import history_evaluation


from sklearn.model_selection import train_test_split

# Otimizadores opcionais
try:
    from tensorflow_addons.optimizers import LAMB
except ImportError:
    LAMB = None


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
    """
    Returns a cosine decay restart learning rate schedule.
    """
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
    Gera um caminho único para o arquivo de checkpoint, incorporando o ID do processo
    e um timestamp.
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
    
    When use_lr_scheduler is True, a ReduceLROnPlateau callback is added.
    """
    callbacks = []
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_loss'
    )
    callbacks.append(model_checkpoint)
    
    if use_lr_scheduler:
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.25,      # Reduce LR by 50% on plateau
            patience=50,     # Wait 20 epochs after improvement stops
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
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
    Retorna uma instância de otimizador com base no optimizer_type.

    Parâmetros:
      optimizer_type: Tipo do otimizador ('adamw', 'adam', 'sgd', 'rmsprop' ou 'adagrad').
      initial_lr: Taxa de aprendizado inicial.
      weight_decay: Decaimento de peso (aplicado apenas no AdamW).
      first_decay_steps: Número de passos para o primeiro ciclo de decaimento da taxa (usado na função de agendamento, se aplicável).
      global_clipnorm: Valor de clipagem global dos gradientes.

    Retorna:
      Uma instância de tf.keras.optimizers.Optimizer configurada conforme o tipo escolhido.
    """
    # Supondo que get_lr_schedule seja uma função que retorna um agendador de taxa de aprendizado com decay cosseno.
    if optimizer_type.lower() == 'adamw':
        lr_schedule = get_lr_schedule(initial_lr, first_decay_steps)
        optimizer = tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            global_clipnorm=global_clipnorm
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_lr,
            clipvalue=global_clipnorm  # clipagem dos gradientes
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=initial_lr,
            momentum=0.0,
            clipnorm=global_clipnorm  # clipagem dos gradientes
        )
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=initial_lr,
            clipnorm=global_clipnorm  # clipagem dos gradientes
        )
    elif optimizer_type.lower() == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=initial_lr,
            clipnorm=global_clipnorm  # clipagem dos gradientes
        )
    else:
        raise ValueError("Tipo de otimizador não suportado. Escolha entre 'adamw', 'adam', 'sgd', 'rmsprop' ou 'adagrad'.")
    
    return optimizer



class HistoryObject:
    def __init__(self):
        self.history = {"loss": [], "val_loss": []}

def train_with_tape(
    model, optimizer, X, y, epochs, batch_size,
    patience, checkpoint_path, diagnostic_interval
):
    train_dataset, val_dataset = create_tf_datasets(X, y, batch_size)
    adaptive_loss_fn = model.loss

    best_val_loss = np.inf
    epochs_no_improve = 0
    history_obj = HistoryObject()  # Use our History-like object

    for epoch in range(epochs):
        tf.print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = []

        for step, (batch_X, batch_y) in enumerate(train_dataset):
            loss = train_step(
                model, adaptive_loss_fn, optimizer, batch_X, batch_y,
                diagnostic_interval, epoch, step
            )
            epoch_loss.append(loss)

        epoch_loss_avg = np.mean(epoch_loss)
        val_loss_avg = evaluate_model(model, adaptive_loss_fn, val_dataset)

        # Record the metrics in our history object
        history_obj.history["loss"].append(epoch_loss_avg)
        history_obj.history["val_loss"].append(val_loss_avg)

        tf.print(f"Epoch loss: {epoch_loss_avg:.4f} | Val loss: {val_loss_avg:.4f}")

        # Early stopping logic
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            model.save_weights(checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tf.print("Early stopping triggered.")
                break

    # Load best model
    model.load_weights(checkpoint_path)
    history_evaluation(history_obj)  # Now works because history_obj has .history
    return model, history_obj


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
    Treina o modelo utilizando GradientTape, reproduzindo o comportamento
    do pipeline tradicional com model.fit, incluindo avaliação, early stopping
    e checkpointing.
    """
    # Cria os datasets de treino e validação com o mesmo pipeline tradicional
    train_ds, val_ds = create_tf_datasets(X, y, batch_size)
    best_val_loss = np.inf
    epochs_no_improve = 0
    history_obj = HistoryObject()  # Reuse your History-like object
    
    # Define a função de treinamento por batch com @tf.function para otimização
    @tf.function
    def train_step(batch_X, batch_y):
        with tf.GradientTape() as tape:
            preds = model(batch_X, training=True)
            # Calcula a perda média para o batch
            loss = tf.keras.losses.MeanAbsoluteError()(batch_y, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        epoch_losses = []
        # Itera sobre os batches do dataset de treino
        for batch_X, batch_y in train_ds:
            loss = train_step(batch_X, batch_y)
            epoch_losses.append(loss)
        # Calcula a perda média do epoch
        avg_epoch_loss = np.mean([loss.numpy() for loss in epoch_losses])
        # Avalia o modelo no dataset de validação
        val_loss = evaluate_model(model, tf.keras.losses.MeanAbsoluteError(), val_ds)
        
        history_obj.history["loss"].append(avg_epoch_loss)
        history_obj.history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Se houve melhoria na validação, salva os pesos e reseta o contador
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_weights(checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Carrega os melhores pesos salvos antes de retornar
    model.load_weights(checkpoint_path)
    return model, history_obj


# =============================================================================
# Análise: Curva de validação, Diversidade e Ensemble dos Snapshots
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.set_loglevel("warning")


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

    training_losses = {
        "out": tf.keras.losses.MeanAbsoluteError(),
        "q_phys": tf.keras.losses.MeanAbsoluteError(),
        "residual": tf.keras.losses.MeanAbsoluteError()
    }
    

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



from common.common import create_internal_validation_set_from_disk

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



# =============================================================================
# Função moderna de Treino para assegurar a convergência
# =============================================================================

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
    cycles: int = 7,
    use_mixed_precision: bool = True,
    validation_mode: str = "hybrid"
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
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
    ]

    # Prepare training and validation sets
    X_train, y_train, X_val_final, y_val_final = _prepare_validation_data(
        X, y, X_val, y_val, validation_split, mode=validation_mode
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
        n_best=8
    )

    model._snapshot_weights = snapshot_weights
    history_evaluation(history)
    evaluate_snapshots_and_ensemble(model, X_val_final, y_val_final, snapshot_weights, snapshot_epochs, history.history)
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
    checkpoint_path: str
) -> tf.keras.Model:
    """
    Train a single block of the hybrid model (Trend or Physics).
    """
    model = create_model(
        **model_args,
        freeze_trend=freeze_trend,
        freeze_physics=freeze_physics,
        fusion_type=fusion_type
    )
    model, _ = train_modern(
        model=model,
        X_train=X_train,
        y_train=y_train,
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
    physics_model: tf.keras.Model
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
    checkpoint_path: str
) -> Tuple[tf.keras.Model, dict]:
    """
    Stage-wise training for hybrid models combining Trend and Physics blocks:
      1. Train Trend block alone.
      2. Train Physics block alone.
      3. Train fusion layer with blocks frozen.
      4. Fine-tune complete model with all layers trainable.
    """
    # Stage 1: Trend block
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
        checkpoint_path='trend_only.keras'
    )

    # Stage 2: Physics block
    physics_model = _train_individual_block(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val, 
        y_val=y_val,
        model_args=model_args,
        freeze_trend=True,
        freeze_physics=False,
        fusion_type='pin',
        epochs=20,
        batch_size=batch_size,
        patience=patience,
        optimizer_type=optimizer_type,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        checkpoint_path='pin_only.keras'
    )

    # Stage 3: Fusion layer training
    fusion_args = {**model_args, 'freeze_trend': True, 'freeze_physics': True}
    fusion_model = _assemble_fusion_model(fusion_args, trend_model, physics_model)
    fusion_model, _ = train_modern(
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
    fusion_model.get_layer('trend_block').trainable = True
    fusion_model.get_layer('physics_block').trainable = True
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
        checkpoint_path=checkpoint_path
    )

    return fusion_model, history

def train_hybrid_three_stages(
    X, y,
    X_val: Optional[Union[np.ndarray, List[np.ndarray]]],
    y_val: Optional[np.ndarray], 
    build_kwargs: dict,
    epochs_per_stage: int,
    batch_size: int,
    patience: int,
    optimizer_cfg: dict,
    val_split: float = 0.1,
    cycles: int = 5,
    use_mixed_precision: bool = True
) -> Tuple[tf.keras.Model, dict]:
    # ——————————————————————————————————————————
    # 1) monta o modelo “dict” com cabeça de extrator
    # ——————————————————————————————————————————
    model = create_model(
        **build_kwargs,
        output_mode='dict',
        add_extractor_head=True
    )

    # garante pasta de checkpoints
    ckpt_dir = "checkpoint"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt1 = f"{ckpt_dir}/s1_physics_best.keras"
    ckpt2 = f"{ckpt_dir}/s2_extractor_best.keras"
    ckpt3 = f"{ckpt_dir}/s3_fuser_combiner_best.keras"

    # extrai parâmetros do optimizador para o train_modern
    opt_cls     = optimizer_cfg
    lr0         = 1e-4
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
        initial_lr=lr0,
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
        initial_lr=lr0,
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
        initial_lr=lr0,
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

from forecast_pipeline.config import DEFAULT_DATASET
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
    initial_lr: float = 1e-2,
    weight_decay: float = 1e-4,
    first_decay_steps: int = 100,
    checkpoint_path: str = 'best_model.keras',
    use_gradient_tape: bool = False,
    training_mode: str = "traditional",
    architecture_name: Optional[str] = None
) -> Tuple[tf.keras.Model, dict]:
    """
    Train the model using traditional, GradientTape, or hybrid staged strategies.
    """
    optimizer = get_optimizer(optimizer_type, initial_lr, weight_decay, first_decay_steps)

    logging.info(f"training_mode: {training_mode}")

    if use_gradient_tape:
        return train_with_tape(
            model, optimizer, X_train, y_train, epochs, batch_size, patience, checkpoint_path
        )

    model_args = {
            "input_shape": X_train,
            "horizon": y_train.shape[1],
            "strategy_config": model.strategy_config,
            "architecture_name": architecture_name,
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
            checkpoint_path
        )

    if architecture_name == "Seq2Fuser":

        model_args = {
            "input_shape": X_train,
            "horizon": y_train.shape[1],
            "strategy_config": model.strategy_config,
            "architecture_name": architecture_name,
            "fuser_config": model.fuser_config,
            "extractor_config": model.extractor_config
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
    training_mode: str = "traditional"
) -> Tuple[tf.keras.Model, Dict, np.ndarray]:
    """
    Unified entrypoint replacing train_single_model:
      - Accepts same parameters for backward compatibility.
      - Trains model and returns predictions on `prediction_input`.
    """
    # Extract and remove optional config dicts
    strategy_config = train_kwargs.get('strategy_config')
    extractor_config = train_kwargs.get('extractor_config')
    fuser_config     = train_kwargs.get('fuser_config')


    # Prepare training data
    X_train = train_kwargs['X_train']
    y_train = train_kwargs['y_train']
    X_val   = train_kwargs.get('X_val')
    y_val   = train_kwargs.get('y_val')

    SEQ2SEQ_ARCHS = ["Seq2Context", "Seq2PIN", "Seq2Trend"]
    if y_train.ndim > 1 and architecture_name in SEQ2SEQ_ARCHS:
        model = create_model(
            input_shape=X_train,
            horizon=y_train.shape[1],
            strategy_config=strategy_config,
            extractor_config=extractor_config,
            fuser_config=fuser_config,
            architecture_name=architecture_name
        )
        model.strategy_config   = strategy_config

    elif y_train.ndim > 1 and architecture_name=="Seq2Fuser":
        model = create_model(
            input_shape=X_train,
            horizon=y_train.shape[1],
            strategy_config=strategy_config,
            extractor_config=extractor_config,
            fuser_config=fuser_config,
            architecture_name=architecture_name
        )
        # 3) ANEXA as configs como atributos do model
        model.strategy_config   = strategy_config
        model.extractor_config  = extractor_config
        model.fuser_config      = fuser_config
    else:
        model = create_model(
            input_shape=X_train,
            architecture_name=architecture_name
        )

    return train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,          
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        training_mode=training_mode,
        architecture_name=architecture_name
    )
