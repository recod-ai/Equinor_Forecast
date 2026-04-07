# src/common/common.py
from __future__ import annotations

from dataclasses import dataclass
import logging
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.evaluation import plot_predictions



# augment_phys.py
# ---------------------------------------------------------------------------
#  Data-augmentation modular para séries temporais de produção (óleo & gás)
#  - Mantém a “redução de escala” do alvo (multiplicativa) mas de forma
#    fisicamente coerente.
#  - Permite ligar/desligar técnicas por flags.
#  - Funciona direto no pipeline (substitui augment_with_synthetic_samples).
# ---------------------------------------------------------------------------


# ------------------------ helpers de consistência --------------------------
def _ensure_ndarray(x) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.values
    return np.asarray(x)


def _concat(orig: np.ndarray, synt: List[np.ndarray], like):
    out = np.concatenate([orig, *synt], axis=0)
    if isinstance(like, (pd.DataFrame, pd.Series)):
        # reconstrói mesmo tipo/índices – assume reset de índice é ok para treino
        return type(like)(out, columns=getattr(like, "columns", None))
    return out



def augment_with_synthetic_samples(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    scales: List[float] = [1.5, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19]
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Augments the training data by scaling the amplitude of features and target,
    working generically with both pandas objects and NumPy arrays.

    Parameters:
      X_train: Original training features (pd.DataFrame or np.ndarray).
      y_train: Original target values (pd.Series or np.ndarray).
      scales: List of scale factors to use.

    Returns:
      A tuple (augmented_X_train, augmented_y_train)
    """
    
    # Lista para armazenar os dados originais e os escalados
    X_synthetic = [X_train]
    y_synthetic = [y_train]
    
    # Aplica a escala para cada fator e acumula os resultados
    for scale in scales:
        X_scaled = X_train / scale
        y_scaled = y_train / scale
        X_synthetic.append(X_scaled)
        y_synthetic.append(y_scaled)
    
    # Concatenando os dados de acordo com o tipo
    if isinstance(X_train, pd.DataFrame):
        X_train_aug = pd.concat(X_synthetic, axis=0).reset_index(drop=True)
    elif isinstance(X_train, np.ndarray):
        X_train_aug = np.concatenate(X_synthetic, axis=0)
    else:
        raise TypeError(f"Tipo não suportado para X_train: {type(X_train)}")
    
    if isinstance(y_train, pd.Series):
        y_train_aug = pd.concat(y_synthetic, axis=0).reset_index(drop=True)
    elif isinstance(y_train, np.ndarray):
        y_train_aug = np.concatenate(y_synthetic, axis=0)
    else:
        raise TypeError(f"Tipo não suportado para y_train: {type(y_train)}")
    
    
    return X_train_aug, y_train_aug



import numpy as np
import pickle
from typing import Tuple

def create_internal_validation_set_from_disk(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    metadata_path: str,
    val_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cria um conjunto de treino e validação interna a partir de dados aumentados
    e metadados salvos em disco, garantindo a separação cronológica dentro de cada bloco.

    Args:
        X_aug: O array completo de features aumentado.
        y_aug: O array completo de rótulos aumentado.
        metadata_path: Caminho para o arquivo .pkl contendo os metadados dos blocos.
        val_frac: Fração de cada bloco a ser usada para validação (ex: 0.1 para 10%).

    Returns:
        Uma tupla contendo (X_train_fit, y_train_fit, X_val_internal, y_val_internal).
    """
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        end_indices = metadata["end_indices"]
    except (FileNotFoundError, IOError) as e:
        raise RuntimeError(f"Não foi possível carregar o arquivo de metadados de '{metadata_path}'. "
                         f"Certifique-se de que a função de aumento foi executada com a flag "
                         f"'save_metadata_path'. Erro original: {e}")

    # Listas para coletar as partes de cada novo conjunto
    X_train_fit_parts = []
    y_train_fit_parts = []
    X_val_internal_parts = []
    y_val_internal_parts = []

    start_index = 0
    for end_index in end_indices:
        # Pega o bloco de dados atual (original ou sintético)
        X_block = X_aug[start_index:end_index]
        y_block = y_aug[start_index:end_index]

        # Calcula o ponto de divisão dentro do bloco atual
        n_block_samples = len(y_block)
        if n_block_samples == 0:
            continue

        split_point = int(n_block_samples * (1 - val_frac))

        # Adiciona os primeiros (1 - val_frac)% ao conjunto de treino
        X_train_fit_parts.append(X_block[:split_point])
        y_train_fit_parts.append(y_block[:split_point])
        
        # Adiciona os últimos val_frac% ao conjunto de validação
        X_val_internal_parts.append(X_block[split_point:])
        y_val_internal_parts.append(y_block[split_point:])
        
        # Atualiza o índice de início para o próximo bloco
        start_index = end_index

    # Concatena todas as partes de treino e validação
    # Lida com o caso em que uma das listas pode estar vazia (se val_frac=0 ou 1)
    X_train_fit = np.concatenate(X_train_fit_parts, axis=0) if X_train_fit_parts else np.array([])
    y_train_fit = np.concatenate(y_train_fit_parts, axis=0) if y_train_fit_parts else np.array([])
    X_val_internal = np.concatenate(X_val_internal_parts, axis=0) if X_val_internal_parts else np.array([])
    y_val_internal = np.concatenate(y_val_internal_parts, axis=0) if y_val_internal_parts else np.array([])

    print(f"Divisão interna criada: "
          f"Treino_fit: {X_train_fit.shape[0]} amostras, "
          f"Val_internal: {X_val_internal.shape[0]} amostras.")

    return X_train_fit, y_train_fit, X_val_internal, y_val_internal




def augment_with_synthetic_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    # --- Parâmetros de Controle Geral ---
    data_sample: float = 0.9,
    random_state: int = 42,
    # --- Novos Parâmetros para Estratégias Avançadas ---
    augmentation_modes: List[str] = ['scale'],
    original_replication_factor: int = 1,
    # --- Parâmetros Específicos do Modo 'scale' ---
    scales: List[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # --- Parâmetros Específicos do Modo 'shift' ---
    shift_std_fraction: float = 0.1,
    num_shift_blocks: int = 5,  # Quantos blocos diferentes de 'shift' gerar
    # --- Parâmetros Específicos do Modo 'mixup' ---
    mixup_alpha: float = 0.2,
    num_mixup_blocks: int = 5, # Quantos blocos diferentes de 'mixup' gerar
    # --- Parâmetro de Metadados ---
    save_metadata_path: Optional[str] = "Meta_validation"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augments training data using a variety of strategies for Seq2Seq models.

    This function generates synthetic data by scaling, shifting, or mixing up
    entire sequences, preserving the internal chronological structure of features
    and the overall chronological order of samples. It also allows replicating
    the original data to give it more weight.

    Args:
        X_train: Original, chronologically ordered training sequences (samples, timesteps, features).
        y_train: Original training target sequences (samples, output_steps).
        data_sample: Fraction of samples to select from each synthetic block.
        random_state: Seed for reproducibility.
        augmentation_modes: List of modes to use. Options: ['scale', 'shift', 'mixup'].
        original_replication_factor: How many times to repeat the original data.
        scales: List of scaling factors for the 'scale' mode.
        shift_std_fraction: For 'shift' mode, the std dev of the random shift is a
                            fraction of the original data's target variable std dev.
        num_shift_blocks: Number of different shifted datasets to generate.
        mixup_alpha: Alpha parameter for the 'mixup' mode interpolation.
        num_mixup_blocks: Number of different mixed-up datasets to generate.
        save_metadata_path: If provided, saves block metadata to this path.

    Returns:
        A tuple containing the new augmented training set and labels.
    """
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
        raise TypeError("X_train and y_train must be numpy arrays.")

    rng = np.random.RandomState(random_state)

    # 1. Replicate Original Data (Implicit Weighting)
    if original_replication_factor < 1:
        raise ValueError("original_replication_factor must be at least 1.")
    X_augmented_list = [X_train] * original_replication_factor
    y_augmented_list = [y_train] * original_replication_factor

    num_original_samples = X_train.shape[0]

    for mode in augmentation_modes:
        if num_original_samples == 0: continue

        # Determine how many synthetic samples to generate per block
        num_to_sample = int(num_original_samples * data_sample)
        if num_to_sample == 0: continue

        if mode == 'scale':
            for scale_factor in scales:
                # This transformation is consistent across all features and time steps
                X_synth_base = X_train / scale_factor
                y_synth_base = y_train / scale_factor

                # Sub-sample chronologically
                chosen_indices = rng.choice(num_original_samples, size=num_to_sample, replace=False)
                sorted_indices = np.sort(chosen_indices)
                X_augmented_list.append(X_synth_base[sorted_indices])
                y_augmented_list.append(y_synth_base[sorted_indices])

        elif mode == 'shift':
            # Calculate a realistic shift based on the data's own variance.
            # Use the std dev of the target variable from the *original* unscaled data.
            target_std = y_train.std()
            shift_std = target_std * shift_std_fraction
            
            for _ in range(num_shift_blocks):
                # For each block, draw a single shift value.
                # This simulates a consistent sensor drift or change in baseline.
                shift_amount = rng.normal(loc=0, scale=shift_std)
                
                # Apply the same shift to all values in X and y.
                # This is physically consistent, assuming all features are related.
                X_synth_base = X_train + shift_amount
                y_synth_base = y_train + shift_amount
                
                # Sub-sample chronologically
                chosen_indices = rng.choice(num_original_samples, size=num_to_sample, replace=False)
                sorted_indices = np.sort(chosen_indices)
                X_augmented_list.append(X_synth_base[sorted_indices])
                y_augmented_list.append(y_synth_base[sorted_indices])

        elif mode == 'mixup':
            for _ in range(num_mixup_blocks):
                # To preserve chronology, we mix a sample with a randomly chosen *other* sample.
                # This creates a "virtual" well with characteristics blended from two real ones.
                mix_indices = rng.choice(num_original_samples, size=num_original_samples, replace=True)
                
                # The mixup is applied at the sample level, preserving internal sequence structure.
                X_synth_base = mixup_alpha * X_train + (1 - mixup_alpha) * X_train[mix_indices]
                y_synth_base = mixup_alpha * y_train + (1 - mixup_alpha) * y_train[mix_indices]

                # Sub-sample chronologically
                chosen_indices = rng.choice(num_original_samples, size=num_to_sample, replace=False)
                sorted_indices = np.sort(chosen_indices)
                X_augmented_list.append(X_synth_base[sorted_indices])
                y_augmented_list.append(y_synth_base[sorted_indices])

        else:
            print(f"Warning: Augmentation mode '{mode}' is not recognized. Skipping.")

    # --- Metadata Logic ---
    if save_metadata_path:
        block_sizes = [len(part) for part in X_augmented_list]
        end_indices = np.cumsum(block_sizes).tolist()
        metadata = {"end_indices": end_indices, "modes_used": augmentation_modes}
        try:
            with open(save_metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        except IOError as e:
            print(f"Error saving metadata to {save_metadata_path}: {e}")

    # --- Final Concatenation ---
    X_final = np.concatenate(X_augmented_list, axis=0)
    y_final = np.concatenate(y_augmented_list, axis=0)

    return X_final, y_final



def split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame em treino e teste, preservando ordem temporal.
    """
    n_test   = int(len(df) * test_size)
    df_train = df.iloc[:-n_test]
    df_test  = df.iloc[-n_test:]
    return df_train, df_test





# =============================================================================
# ONE-BUTTON METHODS (single source of truth)
# =============================================================================

# Canonical families (used by campaign generator)
FAMILY_SEQ2 = "Seq2"
FAMILY_DARTS = "Darts"
FAMILY_ARPS = "Arps"
ALL_FAMILIES = {FAMILY_SEQ2, FAMILY_DARTS, FAMILY_ARPS}

# Canonical method keys (stable public API)
METHOD_ARPS_PURE = "ARPS_PURE"
METHOD_PINN_PURE = "PINN_PURE"
METHOD_DARTS_PURE = "DARTS_PURE"
METHOD_PINN_PLUS_ANALYTIC = "PINN_PLUS_ANALYTIC"
METHOD_ARPS_ENSEMBLE = "ARPS_ENSEMBLE"

ALL_METHODS = {
    METHOD_ARPS_PURE,
    METHOD_PINN_PURE,
    METHOD_DARTS_PURE,
    METHOD_PINN_PLUS_ANALYTIC,
    METHOD_ARPS_ENSEMBLE,
}


@dataclass(frozen=True)
class MethodSpec:
    """
    Defines a single top-level forecasting method (one-button selection).

    - arch: which validation/inference family to route to ("arps" | "seq2" | "darts")
    - pipeline_preset: "OFF" or a valid PIPELINE_PRESET key (e.g., "PINN_PLUS_ANALYTIC")
    - architecture_name: optional (e.g., "Seq2PIN" for seq2 family)
    """
    key: str
    arch: str                 # "arps" | "seq2" | "darts"
    pipeline_preset: str      # "OFF" or a valid PIPELINE_PRESET key
    architecture_name: Optional[str] = None
    description: str = ""


# ---- Canonical method catalog (your 5 modes) ----
METHOD_CATALOG: Dict[str, MethodSpec] = {
    METHOD_ARPS_PURE: MethodSpec(
        key=METHOD_ARPS_PURE,
        arch="arps",
        pipeline_preset="OFF",
        architecture_name=None,
        description="Pure ARPS (single curve). No ensemble. No offline-analytic coupling.",
    ),
    METHOD_PINN_PURE: MethodSpec(
        key=METHOD_PINN_PURE,
        arch="seq2",
        pipeline_preset="OFF",
        architecture_name="Seq2PIN",
        description="Pure PINN (Seq2) inference. No analytic coupling.",
    ),
    METHOD_DARTS_PURE: MethodSpec(
        key=METHOD_DARTS_PURE,
        arch="darts",
        pipeline_preset="OFF",
        architecture_name=None,
        description="Pure Darts inference. No analytic coupling.",
    ),
    METHOD_PINN_PLUS_ANALYTIC: MethodSpec(
        key=METHOD_PINN_PLUS_ANALYTIC,
        arch="seq2",
        pipeline_preset="PINN_PLUS_ANALYTIC",
        architecture_name="Seq2PIN",
        description="PINN + Offline-Analytic coupling (spaghetti) using ARPS coupling logic.",
    ),
    METHOD_ARPS_ENSEMBLE: MethodSpec(
        key=METHOD_ARPS_ENSEMBLE,
        arch="arps",
        pipeline_preset="ARPS_ENSEMBLE_SPAGHETTI",
        architecture_name=None,
        description="Pure ARPS canonical fit + theta sampling spaghetti ensemble (optional trimming + aggregation).",
    ),
}


# =============================================================================
# Resolution helpers (Validation + HPO)
# =============================================================================

def _validate_family(family: str) -> str:
    f = str(family).strip()
    if f not in ALL_FAMILIES:
        raise ValueError(f"Unknown family={f!r}. Choose one of: {sorted(ALL_FAMILIES)}")
    return f


def resolve_method(method: str, *, allow_aliases: bool = True) -> MethodSpec:
    """
    Validation notebook resolver:
      METHOD -> MethodSpec(arch, preset, architecture_name)

    Canonical keys:
      ARPS_PURE | PINN_PURE | DARTS_PURE | PINN_PLUS_ANALYTIC | ARPS_ENSEMBLE

    Aliases (optional):
      "arps" -> ARPS_PURE
      "seq2" / "pinn" -> PINN_PURE
      "darts" -> DARTS_PURE
      "pinn+analytic" -> PINN_PLUS_ANALYTIC
      "arps_ensemble" -> ARPS_ENSEMBLE
    """
    if method is None:
        raise ValueError("METHOD is None. Choose a valid method key.")

    m = str(method).strip()
    key = m.upper()

    if allow_aliases:
        aliases = {
            "ARPS": METHOD_ARPS_PURE,
            "SEQ2": METHOD_PINN_PURE,
            "PINN": METHOD_PINN_PURE,
            "DARTS": METHOD_DARTS_PURE,
            "PINN+ANALYTIC": METHOD_PINN_PLUS_ANALYTIC,
            "ARPS_ENSEMBLE": METHOD_ARPS_ENSEMBLE,
        }
        key = aliases.get(key, key)

    spec = METHOD_CATALOG.get(key)
    if spec is None:
        valid = ", ".join(sorted(METHOD_CATALOG.keys()))
        raise ValueError(f"Unknown METHOD={method!r}. Valid options: {valid}")
    return spec


def summarize_methods() -> str:
    lines = ["Available METHODS:"]
    for k in sorted(METHOD_CATALOG.keys()):
        s = METHOD_CATALOG[k]
        lines.append(f"  - {k}: arch={s.arch}, preset={s.pipeline_preset} :: {s.description}")
    return "\n".join(lines)


# =============================================================================
# HPO: family-aware method plan + job_defaults builder
# =============================================================================

def method_is_compatible_with_family(method: str, family: str) -> bool:
    """
    Campaign generator compatibility:
      - Seq2: PINN_PURE or PINN_PLUS_ANALYTIC
      - Darts: DARTS_PURE
      - Arps: ARPS_PURE or ARPS_ENSEMBLE
    """
    spec = resolve_method(method, allow_aliases=True)
    fam = _validate_family(family)

    if fam == FAMILY_SEQ2:
        return spec.key in (METHOD_PINN_PURE, METHOD_PINN_PLUS_ANALYTIC)
    if fam == FAMILY_DARTS:
        return spec.key in (METHOD_DARTS_PURE,)
    if fam == FAMILY_ARPS:
        return spec.key in (METHOD_ARPS_PURE, METHOD_ARPS_ENSEMBLE)
    return False


def default_fallback_method_for_family(family: str) -> str:
    fam = _validate_family(family)
    if fam == FAMILY_SEQ2:
        return METHOD_PINN_PURE
    if fam == FAMILY_DARTS:
        return METHOD_DARTS_PURE
    if fam == FAMILY_ARPS:
        return METHOD_ARPS_PURE
    return METHOD_PINN_PURE


def resolve_method_for_family(
    *,
    method_by_family: Mapping[str, str],
    family: str,
    policy: str = "error",
) -> MethodSpec:
    """
    HPO resolver:
      METHOD_BY_FAMILY + family -> MethodSpec

    policy:
      - "error": raise if missing/incompatible
      - "fallback": fallback to pure method for that family
    """
    fam = _validate_family(family)
    pol = str(policy).strip().lower()
    if pol not in ("error", "fallback"):
        raise ValueError("policy must be 'error' or 'fallback'")

    raw = method_by_family.get(fam)
    if raw is None:
        if pol == "fallback":
            return resolve_method(default_fallback_method_for_family(fam))
        raise ValueError(f"METHOD_BY_FAMILY is missing an entry for family={fam!r}")

    spec = resolve_method(raw, allow_aliases=True)
    if not method_is_compatible_with_family(spec.key, fam):
        if pol == "fallback":
            return resolve_method(default_fallback_method_for_family(fam))
        raise ValueError(f"METHOD={spec.key!r} is not compatible with family={fam!r}")

    return spec


def build_methods_plan_for_active_families(
    *,
    active_families: List[str],
    method_by_family: Mapping[str, str],
    policy: str = "error",
) -> Dict[str, MethodSpec]:
    """
    Convenience:
      active_families + METHOD_BY_FAMILY -> {family: MethodSpec}
    """
    plan: Dict[str, MethodSpec] = {}
    for fam in active_families:
        f = _validate_family(fam)
        plan[f] = resolve_method_for_family(method_by_family=method_by_family, family=f, policy=policy)
    return plan


# -----------------------------------------------------------------------------
# Deep merge utilities (predictable, no mutation)
# -----------------------------------------------------------------------------
def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionaries:
      - dict + dict => merge keys recursively
      - otherwise => override wins
    Returns a NEW dict.
    """
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = deep_merge_dicts(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# Non-breaking behavior controls
# -----------------------------------------------------------------------------
def minimal_off_overrides() -> Dict[str, Any]:
    """
    Minimal "pure mode" overrides for HPO YAMLs.
    NOTE: we do NOT touch aggregation_* fields to avoid breaking existing assumptions.
    """
    return {
        "latent_mode": "off",
        "latent_cfg": {"mode": "off"},
    }


def sanitize_job_defaults(
    job_defaults: Mapping[str, Any],
    *,
    preserve_legacy_fields: bool = True,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Default is NON-BREAKING:
      preserve_legacy_fields=True keeps any leftover blocks (e.g., latent_cfg keys).

    If you ever want strict cleanup later, set preserve_legacy_fields=False and
    expand this function (Patch 2+).
    """
    out = dict(job_defaults)

    if preserve_legacy_fields:
        return out

    # Optional strict cleanup:
    fam = _validate_family(family) if family is not None else None
    if fam == FAMILY_DARTS:
        out.pop("arps_ensemble", None)

    return out


# -----------------------------------------------------------------------------
# Build per-family job_defaults: base_user + preset(job_defaults) + minimal_off
# -----------------------------------------------------------------------------
def build_job_defaults_effective_for_family(
    *,
    base_user: Mapping[str, Any],
    family: str,
    spec: MethodSpec,
    pipeline_presets: Mapping[str, Any],
    preserve_legacy_fields: bool = True,
) -> Dict[str, Any]:
    """
    Merge order (later wins):
      1) base_user
      2) preset job_defaults (if spec.pipeline_preset != "OFF")
      3) minimal_off_overrides (if spec.pipeline_preset == "OFF")

    We DO NOT set 'architecture_name' here; caller sets it per-architecture.
    """
    fam = _validate_family(family)
    jd = dict(base_user)

    # 2) preset job_defaults (specialized methods)
    if spec.pipeline_preset and str(spec.pipeline_preset).upper() != "OFF":
        preset_name = spec.pipeline_preset
        if preset_name not in pipeline_presets:
            raise ValueError(
                f"Preset {preset_name!r} required by METHOD={spec.key!r} not found in PIPELINE_PRESETS. "
                f"Available: {sorted(pipeline_presets)}"
            )

        preset_obj = pipeline_presets[preset_name]

        # preset_obj can be PipelinePreset dataclass or dict-like
        preset_overrides = getattr(preset_obj, "overrides", None)
        if preset_overrides is None and isinstance(preset_obj, Mapping):
            preset_overrides = preset_obj.get("overrides", None)

        preset_jd: Dict[str, Any] = {}
        if isinstance(preset_overrides, Mapping):
            preset_jd = dict(preset_overrides.get("job_defaults", {}) or {})

        jd = deep_merge_dicts(jd, preset_jd)

    # 3) pure modes: explicitly force latent off (minimal)
    else:
        jd = deep_merge_dicts(jd, minimal_off_overrides())

    # Ensure user knobs win over presets for HPO (non-breaking, explicit)
    USER_KNOBS_THAT_MUST_WIN = {
        "seed",
        "plot",
        "lag_window",
        "horizon",
        "patience",
        "test_size",
        "val_size",
        "feature_kind",
        "use_known_good",
        "evaluate_by_slice",
        "aggregation_quantiles",
        "scenario",
        "band",
        "show_components",
    }
    
    for k in USER_KNOBS_THAT_MUST_WIN:
        if k in base_user:
            jd[k] = base_user[k]


    # non-breaking sanitizer
    jd = sanitize_job_defaults(jd, preserve_legacy_fields=preserve_legacy_fields, family=fam)
    return jd

