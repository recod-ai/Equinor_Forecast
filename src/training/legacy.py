from typing import Union, List, Tuple, Optional
import os, numpy as np, tensorflow as tf

class WarmUpCosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Warm-up linear + CosineDecayRestarts scheduling.
    """
    def __init__(
        self,
        initial_learning_rate: float,
        first_decay_steps: int,
        warmup_steps: int = 0,
        t_mul: float = 1.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        name: str = None,
        dtype=tf.float32,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.warmup_steps = warmup_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.name = name
        self.dtype = dtype
        # CosineDecayRestarts não aceita dtype em algumas versões
        self.cosine = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha,
            name=name,
        )

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        wm_f = tf.cast(self.warmup_steps, tf.float32)
        if self.warmup_steps > 0:
            warmup_lr = self.initial_learning_rate * (step_f / wm_f)
            return tf.cond(
                step_f < wm_f,
                lambda: warmup_lr,
                lambda: self.cosine(step - self.warmup_steps),
            )
        return self.cosine(step)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "warmup_steps": self.warmup_steps,
            "t_mul": self.t_mul,
            "m_mul": self.m_mul,
            "alpha": self.alpha,
            "name": self.name,
            "dtype": self.dtype,
        }





class BatchGrowthScheduler(tf.keras.callbacks.Callback):
    """
    Cresce batch size conforme agendamento de (start_epoch, new_batch_size).
    """
    def __init__(self, batch_schedule: List[tuple], initial_size: int, steps_per_epoch: int, verbose: bool = False):
        super().__init__()
        self.batch_schedule = batch_schedule
        self.current_stage = 0
        self.initial_size = initial_size
        self.steps_per_epoch = steps_per_epoch
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if self.current_stage < len(self.batch_schedule) - 1:
            next_epoch, next_size = self.batch_schedule[self.current_stage + 1]
            if epoch >= next_epoch:
                self.current_stage += 1
                tf.keras.backend.set_value(self.model.optimizer.iterations, 0)
                tf.keras.utils.get_custom_objects()["batch_size"] = next_size
                if self.verbose:
                    print(f"BatchGrowthScheduler: epoch={epoch}, new batch_size={next_size}")


# 1) Função tradicional (com warm-up + cosine e opcional batch growth)
def train_traditionally(
    model: tf.keras.Model,
    X: Union[np.ndarray, List[np.ndarray]],
    y: np.ndarray,
    epochs: int = 300,
    batch_size: int = 32,
    patience: int = 100,
    optimizer_type: str = "adam",
    initial_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    first_decay_steps: int = 50,
    checkpoint_prefix: str = 'best_model.keras',
    val_split: float = 0.1,
    cycles: int = 5,
    warmup_frac: float = 0.1,
):
    """
    Versão aprimorada mas compatível com a assinatura original.
    Extras:
      - warmup_frac: fração do primeiro ciclo dedicada ao warm-up
      - batch_schedule: lista de (start_epoch, batch_size) para crescimento dinâmico
    """
    # forçar tipos
    epochs = int(epochs)
    batch_size = int(batch_size)
    cycles = int(cycles)
    first_decay_steps = int(first_decay_steps)

    # split treino/val
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_split, random_state=42, shuffle=True)
    steps_per_epoch = math.ceil(len(y_tr) / batch_size)

    # definir ciclo
    first_cycle = first_decay_steps if first_decay_steps > 0 else (epochs * steps_per_epoch) // cycles
    warmup_steps = int(warmup_frac * first_cycle)

    # scheduler base
    base_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_cycle,
        t_mul=1.0,
        m_mul=1.0,
        alpha=0.0,
        name=None,
    )
    lr_sched = WarmUpSchedule(initial_lr, base_sched, warmup_steps) if warmup_steps > 0 else base_sched

    # otimizador
    if optimizer_type.lower() == 'lamb' and LAMB is not None:
        opt = LAMB(learning_rate=lr_sched, weight_decay=weight_decay)
    else:
        opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_sched, weight_decay=weight_decay)
    model.compile(optimizer=opt, loss="mae")

    # callbacks
    callbacks = [
        SnapshotSaver(epochs, cycles, steps_per_epoch),
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_best_only=True, monitor='val_loss', verbose=0)
    ]

    batch_schedule = [
            (0,    4),   # usar batch_size=2 a partir da época 0
            (25,   16),   # trocar para batch_size=4 a partir da época 25
            (50,   32),   # trocar para batch_size=8 a partir da época 50
            (75,   64),
        ]

    
    if batch_schedule:
        callbacks.append(BatchGrowthScheduler(batch_schedule, batch_size, steps_per_epoch))

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )
    model._snapshot_weights = [w for w in callbacks[0].best_weights_cycle if w is not None]
    history_evaluation(history)
    return model, history.history


# def train_traditionally(
#     model: tf.keras.Model,
#     X: Union[np.ndarray, List[np.ndarray]],
#     y: np.ndarray,
#     epochs: int = 300,
#     batch_size: int = 32,
#     patience: int = 100,
#     optimizer_type: str = "adam",
#     initial_lr: float = 1e-3,
#     weight_decay: float = 1e-3,
#     first_decay_steps: int = 50,
#     checkpoint_prefix: str = 'best_model.keras',
#     val_split: float = 0.05,
#     cycles=5
# ):
#     """
#     Devolve (model, history.history) — 'model' agora tem o atributo
#     ._snapshot_weights:  List[List[np.ndarray]]
#     """
#     import numpy as np
#     from sklearn.model_selection import train_test_split
#     import tensorflow as tf
#     import math

#     # 1) split
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X, y, test_size=val_split, random_state=42, shuffle=True
#     )

#     # 2) scheduler cosine-restarts
#     steps_per_epoch = math.ceil(len(y_tr) / batch_size)
#     first_cycle_steps = (epochs * steps_per_epoch) // cycles
#     lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
#         initial_learning_rate=initial_lr,
#         first_decay_steps=first_cycle_steps,
#     )

#     opt = tf.keras.optimizers.get(optimizer_type)
#     opt.learning_rate = lr_sched
#     if weight_decay > 0:
#         opt.weight_decay = weight_decay

#     model.compile(optimizer=opt, loss="mae")

#     # 3) callbacks
#     saver = SnapshotSaver(epochs, cycles, steps_per_epoch)
#     early = tf.keras.callbacks.EarlyStopping(patience=patience,
#                                              restore_best_weights=True,
#                                              verbose=0)

#     history = model.fit(
#         X_tr, y_tr,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[saver, early],
#         shuffle=True,
#         verbose=0,
#     )

#     # 4) snapshots → atributo do modelo
#     model._snapshot_weights = [
#         w for w in saver.best_weights_cycle if w is not None
#     ]
#     history_evaluation(history)
#     return model, history.history


# def train_traditionally(
#     model: tf.keras.Model,
#     X: Union[np.ndarray, List[np.ndarray]],
#     y: np.ndarray,
#     epochs: int = 300,
#     batch_size: int = 32,
#     patience: int = 100,
#     optimizer_type: str = 'adam',
#     initial_lr: float = 1e-3,
#     weight_decay: float = 1e-3,
#     first_decay_steps: int = 50,
#     checkpoint_prefix: str = 'best_model.keras',
#     val_split: float = 0.5,
#     random_state: Optional[int] = None
# ) -> Tuple[
#     tf.keras.Model,
#     dict,
#     float,
#     str,
#     Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray]
# ]:
#     """
#     Treina um modelo Keras com split interno de validação, agendamento de LR,
#     early stopping e model checkpointing, sem SWA.

#     Retorna:
#       - model: modelo carregado com pesos do melhor checkpoint
#       - history: dicionário de métricas por época
#       - val_loss: perda MAE no conjunto de validação
#       - best_ckpt: caminho para o checkpoint salvo
#       - (X_val, y_val): fatia de validação usada
#     """
#     # 1) Split randômico mantendo múltiplos tensores coerentes
#     from sklearn.model_selection import train_test_split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y,
#         test_size=val_split,
#         random_state=random_state,
#         shuffle=True
#     )

#     # 2) Otimizador e compile
#     optimizer = get_optimizer(optimizer_type, initial_lr, weight_decay, first_decay_steps)
#     if model.name != 'pin_1':
#         model.compile(optimizer=optimizer, loss='mae')

#     # 3) Callbacks
#     checkpoint_dir = os.path.join('checkpoint', unique_checkpoint_path(checkpoint_prefix))
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     best_ckpt = os.path.join(checkpoint_dir, 'best.keras')

#     callbacks = get_callbacks(patience, best_ckpt, use_lr_scheduler=True)

#     # 4) Fit com validation_data explícito
#     history = model.fit(
#         X_train,
#         y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks,
#         verbose=0,
#         shuffle=True
#     )

#     # 5) Carregar melhor modelo e avaliar
#     model.load_weights(best_ckpt)
#     val_loss = model.evaluate(X_val, y_val, verbose=0)

#     logging.info(f"Val Mean MAE: {np.mean(val_loss)}")
#     history_evaluation(history)
#     return model, history.history




# def train_traditionally(
#     model: tf.keras.Model,
#     X: Union[np.ndarray, List[np.ndarray]],
#     y: np.ndarray,
#     epochs: int = 300,
#     batch_size: int = 32,
#     patience: int = 100,
#     optimizer_type: str = "adam",
#     initial_lr: float = 1e-3,
#     weight_decay: float = 1e-3,
#     first_decay_steps: int = 50,
#     checkpoint_prefix: str = "model",
#     swa_last_n: int = 5,
#     val_split: float = 0.1,
#     random_state: Optional[int] = None,
# ) -> Tuple[
#         tf.keras.Model,           # modelo já com pesos do SWA
#         dict,                     # history.history
#         float,                    # val_mae pós-SWA
#         str,                      # caminho do best checkpoint
#         Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray]  # (X_val, y_val)
# ]:
#     """
#     Treina o modelo, separa automaticamente validação, faz checkpoints + SWA
#     e devolve: modelo_swa, history, val_mae_swa, best_ckpt, (X_val, y_val).
#     """
#     # ------------------------------------------------------------------ #
#     # 0) Split train / val mantendo coerência entre múltiplos tensores
#     # ------------------------------------------------------------------ #
#     from sklearn.model_selection import train_test_split
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y,
#         test_size=val_split,
#         random_state=15,
#         shuffle=True
#     )

#     # ------------------------------------------------------------------ #
#     # 2) Compila
#     # ------------------------------------------------------------------ #
#     opt = get_optimizer(optimizer_type, initial_lr, weight_decay, first_decay_steps)
#     model.compile(optimizer=opt, loss="mae")

#     # ------------------------------------------------------------------ #
#     # 3) Callbacks para checkpoints / early-stopping
#     # ------------------------------------------------------------------ #
#     ckpt_dir = os.path.join("checkpoints", unique_checkpoint_path(checkpoint_prefix))
#     os.makedirs(ckpt_dir, exist_ok=True)
#     best_ckpt = os.path.join(ckpt_dir, "best.keras")
#     epoch_ckpt = os.path.join(ckpt_dir, "epoch_{epoch:03d}.keras")

#     cbs = [
#         ModelCheckpoint(best_ckpt, save_best_only=True, monitor="val_loss", verbose=0),
#         ModelCheckpoint(epoch_ckpt, save_best_only=False, save_freq="epoch", verbose=0),
#         EarlyStopping(patience=patience, restore_best_weights=False, verbose=0),
#     ]

#     # ------------------------------------------------------------------ #
#     # 4) Treino
#     # ------------------------------------------------------------------ #
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=cbs,
#         shuffle=True,
#         verbose=0,
#     )

#     # ------------------------------------------------------------------ #
#     # 5) Stochastic Weight Averaging: best + últimos N
#     # ------------------------------------------------------------------ #
#     ckpts = [best_ckpt] + [
#         epoch_ckpt.format(epoch=e) for e in range(epochs - swa_last_n + 1, epochs + 1)
#     ]

#     swa_w, n_used = None, 0
#     for p in ckpts:
#         if not os.path.exists(p):        # nem todo checkpoint vai existir
#             continue
#         model.load_weights(p)
#         w = model.get_weights()
#         if swa_w is None:
#             swa_w = [np.array(layer, dtype=np.float64) for layer in w]
#         else:
#             for i, layer in enumerate(w):
#                 swa_w[i] += layer
#         n_used += 1

#     if n_used:
#         swa_w = [layer / n_used for layer in swa_w]
#         model.set_weights(swa_w)

#     # ------------------------------------------------------------------ #
#     # 6) Métrica de validação após SWA
#     # ------------------------------------------------------------------ #
#     # val_mae_swa = model.evaluate(X_val, y_val, verbose=0)
#     # logging.info(f"Validation Mean MAE (SWA): {np.mean(val_mae_swa)}")

#     # ------------------------------------------------------------------ #
#     # history_evaluation(history)
#     return model, history.history


# def train_traditionally(
#     model: tf.keras.Model,
#     X_train: Union[np.ndarray, List[np.ndarray]],
#     y_train: np.ndarray,
#     epochs: int = 300,
#     batch_size: int = 32,
#     patience: int = 100,
#     optimizer_type: str = 'adam',
#     initial_lr: float = 1e-3,
#     weight_decay: float = 1e-3,
#     first_decay_steps: int = 25,
#     checkpoint_prefix: str = 'model',
#     bootstrap: bool = True,
#     swa_last_n: int = 10
# ) -> Tuple[tf.keras.Model, dict]:
#     """
#     Train a Keras model with bootstrap sampling, multiple checkpoints, and SWA.

#     1) Optionally bootstrap the training set.
#     2) Save the best checkpoint and every epoch.
#     3) After training, average the weights from the best + last N epochs (SWA).
#     4) Return the SWA model and the fit history.
#     """
#     # 1) Bootstrap sampling
#     if bootstrap:
#         n = y_train.shape[0]
#         idx = np.random.choice(n, size=n, replace=True)
#         if isinstance(X_train, list):
#             X_bs = [x[idx] for x in X_train]
#         else:
#             X_bs = X_train[idx]
#         y_bs = y_train[idx]
#     else:
#         X_bs, y_bs = X_train, y_train

#     # 2) Compile
#     optimizer = get_optimizer(optimizer_type, initial_lr, weight_decay, first_decay_steps)
#     model.compile(optimizer=optimizer, loss='mae')

#     # 3) Callbacks: best + per-epoch + early stopping
#     ckpt_dir = os.path.join('checkpoints', unique_checkpoint_path(checkpoint_prefix))
#     os.makedirs(ckpt_dir, exist_ok=True)
#     best_ckpt = os.path.join(ckpt_dir, 'best.keras')
#     epoch_pattern = os.path.join(ckpt_dir, 'epoch_{epoch:03d}.keras')

#     callbacks = [
#         ModelCheckpoint(best_ckpt, save_best_only=True, monitor='val_loss', verbose=0),
#         ModelCheckpoint(epoch_pattern, save_best_only=False, save_freq='epoch', verbose=0),
#         EarlyStopping(patience=patience, restore_best_weights=False, verbose=0)
#     ]

#     # 4) Fit
#     history = model.fit(
#         X_bs, y_bs,
#         validation_split=0.1,
#         # validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=callbacks,
#         verbose=0,
#         shuffle=True
#     )

#     # 5) Stochastic Weight Averaging (SWA)
#     # Collect checkpoints: best + last swa_last_n epochs
#     ckpts = [best_ckpt]
#     last_epochs = list(range(epochs - swa_last_n + 1, epochs + 1))
#     ckpts += [epoch_pattern.format(epoch=e) for e in last_epochs]

#     # Initialize SWA weights sum
#     swa_weights = None
#     count = 0
#     for path in ckpts:
#         if not os.path.exists(path):
#             continue
#         model.load_weights(path)
#         w = model.get_weights()
#         if swa_weights is None:
#             swa_weights = [np.array(layer_w, dtype=np.float64) for layer_w in w]
#         else:
#             for i, lw in enumerate(w):
#                 swa_weights[i] += lw
#         count += 1

#     if count > 0:
#         swa_weights = [layer_w / count for layer_w in swa_weights]
#         model.set_weights(swa_weights)

#     history_evaluation(history)

#     best_val = min(history.history['val_loss'])
#     logging.info((f"Best validation MAE: {best_val}"))

#     history_evaluation(history)

#     return model, history