# forecast_pipeline/analytics.py
"""
Utility functions for scenarios, envelopes, Monte Carlo, and accumulation.
Prevents notebooks from reimplementing the same logic.
"""
# 1. Future imports
from __future__ import annotations

# 2. Standard library imports
import logging
from typing import Any, Dict, List, Optional, Tuple

# 3. Third-party imports
import numpy as np
import pandas as pd

# 4. Local application imports
from common.seq_preprocessing import aggregate_predictions, reconstruct_true_series
from evaluation.evaluation import evaluate, evaluate_and_plot, evaluate_return_dict
from utils.utilities import _inverse_transform_1d, _looks_scaled
from forecast_pipeline.logging_utils import get_logger, phase, log_context, box_log

# 5. Optional dependencies
try:
    # scipy is optional to avoid a hard dependency; if unavailable,
    # we fall back to a table of common z-scores.
    from scipy.stats import norm
except ImportError:
    norm = None

# Public API definition
__all__ = [
    "scenario_curve",
    "make_envelope",
    "mc_sample",
    "cumulate",
    "evaluate_job",
]


# ---------------------------------------------------------------------
# helpers --------------------------------------------------------------
# ---------------------------------------------------------------------
_Z_CACHE = {}


def _z_score(p: float) -> float:
    """Retorna z tal que Φ(z) = p (p between 0 and 1)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    if p in _Z_CACHE:
        return _Z_CACHE[p]
    if norm is not None:
        z = float(norm.ppf(p))
    else:  # tabela curta para p típicos
        table = {0.90: 1.2815516, 0.10: -1.2815516,
                 0.95: 1.6448536, 0.05: -1.6448536,
                 0.975: 1.9599639, 0.025: -1.9599639}
        if p not in table:
            raise RuntimeError("scipy missing and p not in fallback table")
        z = table[p]
    _Z_CACHE[p] = z
    return z


# ---------------------------------------------------------------------
# API pública ----------------------------------------------------------
# ---------------------------------------------------------------------

def scenario_curve(mu: np.ndarray, sigma: Optional[np.ndarray], p: float) -> np.ndarray:
    """Calcula a curva do percentil *p*.

    *Se ``sigma`` existir*:  μ + z σ.
    Caso contrário assume que ``mu`` é um *stack* de curvas (S,B,H)
    e devolve o percentil empírico ao longo do eixo 0.
    """
    z = _z_score(p)

    if sigma is not None:
        return mu + z * sigma

    # sem σ → usamos percentil dos snapshots
    if mu.ndim < 3:
        raise ValueError("Without sigma, mu must be shape (S,B,H)")
    return np.percentile(mu, p * 100.0, axis=0)

def _to_series(arr: np.ndarray) -> np.ndarray:
    """Converte (B,H) → série (L,) se preciso; caso contrário devolve 1-D."""
    return reconstruct_true_series(arr) if arr.ndim == 2 else arr.ravel()

def make_envelope(mu: np.ndarray,
                  sigma: Optional[np.ndarray],
                  p_lower: float,
                  p_upper: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna curvas inferior / superior (percentis) **já no mesmo comprimento**
    da série agregada.
    """
    # converte janelas → série longa
    mu_series = _to_series(mu)
    sig_series = _to_series(sigma) if sigma is not None else None

    lo = scenario_curve(mu_series, sig_series, p_lower)
    up = scenario_curve(mu_series, sig_series, p_upper)
    return lo, up



def mc_sample(mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
    """Gera *n* amostras Monte Carlo ~(μ, σ²).

    Retorna array (n,B,H).
    """
    if sigma is None:
        raise ValueError("mc_sample requires sigma array")
    if n <= 0:
        raise ValueError("n must be positive")
    eps = np.random.randn(n, *mu.shape)  # (n,B,H)
    return mu + eps * sigma


def cumulate(sequence: np.ndarray) -> np.ndarray:
    """Cumsum ao longo do último eixo (H)."""
    return np.cumsum(sequence, axis=-1)



def build_tags_and_label(_params: Dict[str, Any], _well: str) -> Tuple[Dict[str, Any], str, bool, str]:
        """Assemble human-friendly tags/label and Darts info (no side effects)."""
        arch = _params.get("architecture_name", "")
        is_darts = isinstance(arch, str) and arch.startswith("Darts_")
        darts_model = arch.split("_", 1)[1] if is_darts and "_" in arch else None

        base_tags = {
            "Method": arch,
            "Well": _well,
            "strategy": (
                darts_model
                if is_darts
                else _params.get("strategy_config", {}).get("strategy_name", _params.get("physics_strategy", "N/A"))
            ),
            "extractor": (_params.get("extractor_config", {}) or {}).get("type", "none")
                         if arch in {"Seq2Context", "Seq2Fuser"} else "none",
            "fuser":     (_params.get("fuser_config", {}) or {}).get("type", "none")
                         if arch in {"Seq2Context", "Seq2Fuser"} else "none",
        }

        if arch == "Seq2Context":
            label = (
                f"Well {_well} │ PINN: {base_tags['strategy'].replace('_',' ').title()} │ <br> "
                f"Data-Driven: {base_tags['extractor'].upper()} & {base_tags['fuser'].capitalize()}"
            )
        elif arch == "Seq2PIN":
            label = f"Well {_well} │ PINN: {base_tags['strategy'].replace('_',' ').title()}"
        elif arch == "Seq2Trend":
            label = f"Well {_well} │ PINN + Trend: {base_tags['strategy'].replace('_',' ').title()}"
        elif arch == "Arps_Canonical":
            # A variante (ex: 'hyperbolic') é passada nos hiperparâmetros
            variant = _params.get("variant", "Unknown").capitalize()
            label = f"Well {_well} | Arps: {variant}"
        elif is_darts:
            label = f"Well {_well} │ Darts: {darts_model}"
        else:
            label = f"Well {_well} │ {arch or 'Model'}"

        return base_tags, label, is_darts, (darts_model or "")


def evaluate_model_arps(
    y_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler_target,
    config: Dict[str, Any],
    set_name: str,
    params: Dict[str, Any],
    plot: bool,
    *,
    split_name: str,  # "val" | "test"
) -> Dict[str, Any]:
    """
    Avaliação agregada 1D (série contínua) para ARPS — idêntica ao point-forecast padrão,
    mas devolve também as séries físicas para uso no cumulativo ARPS-only.
    """
    # 1) inverter para físico
    y_true_phys = _inverse_transform_1d(scaler_target, y_scaled)
    y_pred_phys = _inverse_transform_1d(scaler_target, y_pred_scaled)

    # 2) métricas agregadas (físico)
    met_agg = _compute_metrics(y_true_phys, y_pred_phys)

    # 3) plot opcional (reuse do plotter padrão)
    if plot:
        from forecast_pipeline.plotting import _plot_seq
        label = f"{set_name}"
        _plot_seq(
            truth=y_true_phys, pred=y_pred_phys, metrics=met_agg,
            label=label, scaler_target=scaler_target, params=params,
            well=config["wells"][0], is_cum=False, split=split_name,
            plot=True, ensemble_out=None
        )

    return {
        "agg_y_true_phys": y_true_phys,
        "agg_y_pred_phys": y_pred_phys,
        "global_metrics":  met_agg,
    }


def _safe_to_list(x):
    """Return a JSON-serializable list from numpy/pandas/python sequences."""
    try:
        return x.tolist()
    except Exception:
        try:
            return list(x)
        except Exception:
            return [x]

def _extract_split_timestamps(scaler_target, split_name: str, length: int):
    """
    Try to fetch {split}_timestamps from scaler_target._split_ctx.
    Fallback to range(length) if not present.
    """
    split_ctx = getattr(scaler_target, "_split_ctx", {}) or {}
    key = f"{split_name}_timestamps"
    t = split_ctx.get(key)
    if t is None:
        return list(range(int(length)))
    try:
        return list(t)
    except Exception:
        return list(range(int(length)))


def coerce_str_list(x: Any, *, allow_csv: bool = True) -> List[str]:
    """
    Coerce x into a List[str].

    Accepts:
      - list/tuple/set of strings
      - single string:
          - python literal list: "['a','b']"
          - json list: '["a","b"]'
          - csv-ish: "a,b" (optional)
          - single token: "a"
      - None -> []
    """
    if x is None:
        return []

    # already a sequence (but not a string)
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            if v is None:
                continue
            out.append(str(v).strip())
        return out

    # string cases
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []

        # try python literal first: "['a','b']"
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, set)):
                    return [str(i).strip() for i in v if i is not None]
            except Exception:
                pass

        # try json list: '["a","b"]'
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return [str(i).strip() for i in v if i is not None]
            except Exception:
                pass

        # csv fallback: "a,b,c"
        if allow_csv and ("," in s) and ("[" not in s and "]" not in s):
            return [p.strip() for p in s.split(",") if p.strip()]

        # single token
        return [s]

    # anything else -> single token string
    return [str(x).strip()]




def evaluate_job(
    y_test_scaled: np.ndarray,
    y_test_pred:  np.ndarray,
    y_val_scaled: np.ndarray,
    y_val_pred:   np.ndarray,
    scaler_target,
    y_train_original: np.ndarray,
    params: Dict[str, Any],
    config: Dict[str, Any],
    well: str,
    plot: bool = True,
    *,
    ensemble_out: Optional["EnsembleOutput"] = None,
    x_train_main_windows: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any],
           pd.DataFrame, pd.DataFrame, Dict[str, Any],
           Dict[str, Any]]:

    # keep original imports exactly as they are used elsewhere
    from forecast_pipeline._plotting_core import _plot_seq, _plot_integrated_view_from_agg

    def _is_seq2seq_path(_arch: str, _y_test_scaled: np.ndarray) -> Tuple[bool, str]:
        """Decide evaluation path and reason (string kept for logging)."""
        seq2seq_archs = globals().get("SEQ2SEQ_ARCHS", {"Seq2Context", "Seq2Trend", "Seq2PIN"})
        is_seq2seq = (_y_test_scaled.ndim == 2) or (_arch in seq2seq_archs)
        path_reason = "by_arch" if (_arch in seq2seq_archs) else "by_shape"
        return is_seq2seq, path_reason

    def _arps_prebranch_if_needed(_arch: str) -> None:
        """Run the ARPS-only extra branch when applicable (mirrors original behavior)."""
        is_arps_local = isinstance(_arch, str) and _arch.startswith("Arps_")
        if not is_arps_local:
            return

        with phase(logger, "evaluate_arps_point_and_cumulative"):
            # (A) aggregated (point-forecast) in physical scale
            res_val_agg = evaluate_model_arps(
                y_scaled=y_val_scaled,
                y_pred_scaled=y_val_pred,
                scaler_target=scaler_target,
                config=config,
                set_name=label,
                params=params,
                plot=plot,
                split_name="val",
            )
            res_test_agg = evaluate_model_arps(
                y_scaled=y_test_scaled,
                y_pred_scaled=y_test_pred,
                scaler_target=scaler_target,
                config=config,
                set_name=label,
                params=params,
                plot=plot,
                split_name="test",
            )

            # (B) cumulative ARPS-only
            _ = evaluate_cumulative_arps(
                y_true_phys=res_val_agg["agg_y_true_phys"],
                y_pred_phys=res_val_agg["agg_y_pred_phys"],
                y_train_windows=y_train_original,
                x_train_main_windows=x_train_main_windows,
                scaler_target=scaler_target,
                config=config,
                set_name="Cumulative",
                params=params,
                plot=plot,
            )
            _ = evaluate_cumulative_arps(
                y_true_phys=res_test_agg["agg_y_true_phys"],
                y_pred_phys=res_test_agg["agg_y_pred_phys"],
                y_train_windows=y_train_original,
                x_train_main_windows=x_train_main_windows,
                scaler_target=scaler_target,
                config=config,
                set_name="Cumulative",
                params=params,
                plot=plot,
            )

            # (C) (optional) pack into pipeline-shaped DFs — same as original block
            agg_test_df_local = pd.DataFrame([_metrics_row(
                metrics=res_test_agg["global_metrics"], well=well, method=arch,
                category="Aggregated", kind="Series", extra_tags=base_tags
            )])
            agg_val_df_local  = pd.DataFrame([_metrics_row(
                metrics=res_val_agg["global_metrics"],  well=well, method=arch,
                category="Aggregated", kind="Series", extra_tags=base_tags
            )])
            _ = agg_test_df_local, agg_val_df_local  # intentionally unused (mirrors original)

    

    def _seq2seq_branch() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any],
                               pd.DataFrame, pd.DataFrame, Dict[str, Any],
                               Dict[str, Any]]:
        """
        Seq2seq evaluation with optional aggregation sweep.
    
        Plug-and-play update:
          - Keeps OFF preset behavior intact (latent_mode="off" -> legacy eval path).
          - Supports full-sequence evaluation when:
              * latent_mode == "full_sequence", OR
              * predictions are "fullseq-like" (H_pred == split_recon_lengths[split] != horizon).
          - Integrated-view spaghetti members (visual-only):
              * Prefer members provided upstream via ensemble_out.meta:
                    meta["integrated_view_val_members_scaled"]
                    meta["integrated_view_test_members_scaled"]
                These are expected to come from offline_analytic coupling_spaghetti plumbing.
              * Only if those are missing, optionally fall back to selecting last-K windows
                (fullseq_k) from the prediction ribbon (legacy fallback).
          - Members are injected BEFORE _plot_integrated_view_from_agg, and also stored
            into series_artifacts["{split}"]["members"] when available (visual-only).
        """
        # ------------------------------------------------------------------
        # 0) Latent / extrapolation context (read-only, logging + diagnostics)
        # ------------------------------------------------------------------
        latent_cfg: Dict[str, Any] = {}
        split_recon_lengths: Dict[str, Any] = {}
    
        meta: Dict[str, Any] = {}
        if ensemble_out is not None and getattr(ensemble_out, "meta", None):
            meta = ensemble_out.meta or {}
            latent_cfg = (meta.get("latent_cfg") or {}).copy()
            split_recon_lengths = (meta.get("split_recon_lengths") or {}).copy()
    
        # allow overriding / definition via params if ever set there
        if not latent_cfg:
            latent_cfg = (params.get("latent_cfg") or {}).copy()
    
        latent_mode = str(latent_cfg.get("mode", latent_cfg.get("latent_mode", "off"))).strip().lower()
        val_recon_len = split_recon_lengths.get("val")
        test_recon_len = split_recon_lengths.get("test")
    
        logger.info(
            "latent_ctx(evaluate_job.seq2seq) mode=%s split_recon_lengths val=%s test=%s",
            latent_mode,
            val_recon_len,
            test_recon_len,
        )
    
        # keep latent context attached to params for debugging / post-analysis
        params.setdefault("_latent_ctx", {})
        params["_latent_ctx"].update({
            "mode": latent_mode,
            "cfg": latent_cfg,
            "split_recon_lengths": split_recon_lengths,
        })
    
        # ------------------------------------------------------------------
        # 0.1) Full-sequence eval config (used only if we choose fullseq path)
        # ------------------------------------------------------------------
        fullseq_mode = (
            params.get("fullseq_mode")
            or latent_cfg.get("fullseq_mode")
            or "deploy_left_k"
        )
        fullseq_mode = str(fullseq_mode).strip().lower()
    
        raw_k = params.get("fullseq_k", latent_cfg.get("fullseq_k", 1))
        try:
            fullseq_k = int(raw_k)
        except Exception:
            fullseq_k = 1
        if fullseq_k <= 0:
            fullseq_k = 1
    
        fullseq_agg_kind = (
            params.get("fullseq_agg_kind")
            or latent_cfg.get("fullseq_agg_kind")
            or "mean"
        )
        fullseq_agg_kind = str(fullseq_agg_kind).strip().lower()
    
        logger.info(
            "fullseq_k_eval_config mode=%s K=%d agg_kind=%s",
            fullseq_mode, fullseq_k, fullseq_agg_kind,
        )
    
        # ------------------------------------------------------------------
        # 0.5) Thin routing helper for Seq2* evaluation
        # ------------------------------------------------------------------
        def _eval_seq2_for_split(
            split_name: str,
            y_scaled: np.ndarray,
            y_pred_scaled: np.ndarray,
            aggregation_method_local: str,
        ) -> Dict[str, Any]:
            """
            Per-split Seq2* evaluation router.
    
            Modes:
              - Pure legacy:
                    latent_mode != "full_sequence"
                    AND predictions use the training horizon (H_pred == horizon)
                    → classic evaluate_model_seq (sliding windows + warm policies).
    
              - Explicit full-sequence:
                    latent_mode == "full_sequence"
                    → evaluate_fullseq_k_mode, using split_recon_lengths[split]
                      as target_length.
    
              - Full-sequence-like (offline analytic or other post-processing):
                    H_pred == split_recon_lengths[split] != horizon
                    → treated as full-sequence, even if latent_mode == "off".
            """
            # Normalize arrays for shape introspection (do not mutate caller references)
            y_scaled_local = np.asarray(y_scaled, dtype=float)
            y_pred_scaled_local = np.asarray(y_pred_scaled, dtype=float)
    
            # Training horizon (what the model was originally trained with)
            H_train = int(params.get("horizon", y_pred_scaled_local.shape[1]))
            H_pred = int(y_pred_scaled_local.shape[1])
    
            raw_target = split_recon_lengths.get(split_name)
            target_length: Optional[int] = None
            if raw_target is not None:
                try:
                    target_length = int(raw_target)
                except Exception:
                    logger.warning(
                        "latent_ctx(seq2eval) split=%s invalid target_length=%r; "
                        "ignoring and falling back to legacy where needed.",
                        split_name, raw_target,
                    )
                    target_length = None
    
            # Detect “full-sequence-like” predictions:
            is_fullseq_like = (
                target_length is not None
                and target_length > 0
                and H_pred == target_length
                and H_pred != H_train
            )
    
            if is_fullseq_like:
                logger.info(
                    "latent_ctx(seq2eval) split=%s detected_fullseq_like=True "
                    "(latent_mode=%s H_pred=%d H_train=%d H_target=%d)",
                    split_name, latent_mode, H_pred, H_train, target_length,
                )
    
            # Decide whether to use the full-sequence evaluator
            use_full_seq = (latent_mode == "full_sequence") or is_fullseq_like
    
            # Guard: if someone set full_sequence but forgot split_recon_lengths
            if use_full_seq and target_length is None:
                logger.warning(
                    "latent_ctx(seq2eval) split=%s requested full-sequence path "
                    "(latent_mode=%s or fullseq_like) but target_length is missing; "
                    "falling back to legacy evaluate_model_seq.",
                    split_name, latent_mode,
                )
                use_full_seq = False
    
            if use_full_seq:
                train_size = 1.0 - params["test_size"] - params["val_size"]
    
                logger.info(
                    "latent_ctx(seq2eval) split=%s use_full_seq=True "
                    "(latent_mode=%s fullseq_like=%s H_train=%d H_pred=%d "
                    "target_length=%s fullseq_mode=%s fullseq_k=%d)",
                    split_name,
                    latent_mode,
                    str(is_fullseq_like),
                    H_train,
                    H_pred,
                    str(target_length),
                    fullseq_mode,
                    fullseq_k,
                )
    
                return evaluate_fullseq_k_mode(
                    mode=fullseq_mode,
                    K=fullseq_k,
                    split_name=split_name,              # "val" | "test"
                    y_split_scaled=y_scaled_local,
                    y_pred_split_scaled=y_pred_scaled_local,
                    scaler_y=scaler_target,
                    input_length=params["lag_window"],
                    output_length=params["horizon"],
                    train_size=train_size,
                    config=config,
                    eval_title="Seq-to-Seq",
                    set_name=label,
                    ensemble_out=ensemble_out,
                    target_length=target_length,
                    agg_kind=fullseq_agg_kind,
                    plot=False,
                )
    
            # Legacy path (unchanged behavior)
            return evaluate_model_seq(
                y_test_scaled=y_scaled_local,
                y_pred_scaled=y_pred_scaled_local,
                scaler_y=scaler_target,
                input_length=params["lag_window"],
                output_length=params["horizon"],
                train_size=1.0 - params["test_size"] - params["val_size"],
                config=config,
                eval_title="Seq-to-Seq",
                set_name=label,
                aggregation_method=aggregation_method_local,
                quantiles=params.get("aggregation_quantiles"),
                plot=False,
                ensemble_out=ensemble_out,
                split_name=split_name,
            )
    
        # ------------------------------------------------------------------
        # 0.6) Integrated-view spaghetti members helpers (visual-only)
        # ------------------------------------------------------------------
        def _get_attr_or_key(obj: Any, key: str, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
    
        def _to_2d(a: Any) -> Optional[np.ndarray]:
            if a is None:
                return None
            arr = np.asarray(a, dtype=float)
            if arr.size == 0:
                return None
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                return None
            return arr
    
        def _inverse_members_scaled(members_scaled_2d: np.ndarray, scaler) -> np.ndarray:
            """
            Inverse-transform (K, H) to physical units.
            Supports scaler with n_features_in_ == 1 or == H.
            """
            if scaler is None:
                return members_scaled_2d
    
            K, Hm = members_scaled_2d.shape
            nfi = getattr(scaler, "n_features_in_", None)
    
            try:
                if nfi == 1:
                    flat = members_scaled_2d.reshape(-1, 1)
                    inv = scaler.inverse_transform(flat).reshape(K, Hm)
                    return inv
                if isinstance(nfi, (int, np.integer)) and int(nfi) == int(Hm):
                    return scaler.inverse_transform(members_scaled_2d)
                # Fallback (best effort)
                flat = members_scaled_2d.reshape(-1, 1)
                inv = scaler.inverse_transform(flat).reshape(K, Hm)
                return inv
            except Exception:
                logger.debug("integrated_view: inverse_transform failed; keeping scaled members.")
                return members_scaled_2d
    
        def _crop_members_to_len(members_2d: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
            if members_2d is None or target_len <= 0:
                return None
            K, Hm = members_2d.shape
            trim = min(Hm, target_len)
            if trim <= 0:
                return None
            return members_2d[:, :trim]
    
        def _is_fullseq_like_for_split(split_name: str, y_pred_scaled_any: np.ndarray) -> bool:
            """
            Conservative: "fullseq-like" means prediction horizon equals the split reconstruction length
            and differs from training horizon. This is the same logic used by the evaluator router.
            """
            y_pred_scaled_any = np.asarray(y_pred_scaled_any, dtype=float)
            if y_pred_scaled_any.ndim != 2 or y_pred_scaled_any.shape[1] <= 0:
                return False
            H_train = int(params.get("horizon", y_pred_scaled_any.shape[1]))
            H_pred = int(y_pred_scaled_any.shape[1])
            raw_target = split_recon_lengths.get(split_name)
            try:
                target_length = int(raw_target) if raw_target is not None else None
            except Exception:
                target_length = None
            return (
                target_length is not None
                and target_length > 0
                and H_pred == target_length
                and H_pred != H_train
            )
    
        def _select_windows_for_plot_fallback(split_name: str, y_pred_split_scaled: np.ndarray) -> Optional[np.ndarray]:
            """
            Fallback-only: select windows like the fullseq evaluator would:
              - deploy_left_k: try ensemble_out.pred_{split}_left; else use split preds
              - deploy_split_k: use split preds
            Then take the LAST K windows.
            """
            y_pred_split_scaled = _to_2d(y_pred_split_scaled)
            if y_pred_split_scaled is None:
                return None
    
            source = y_pred_split_scaled
            if fullseq_mode == "deploy_left_k":
                left_key = f"pred_{split_name}_left"
                left = _to_2d(_get_attr_or_key(ensemble_out, left_key, None))
                if left is not None and left.shape[1] == source.shape[1]:
                    source = left
    
            K_eff = max(1, int(fullseq_k))
            if source.shape[0] <= K_eff:
                return source
            return source[-K_eff:, :]
    
        # ------------------------------------------------------------------
        # 1) Aggregation policy (single vs sweep)
        # ------------------------------------------------------------------
        agg_method_in = params.get("aggregation_method", "reconstruct")
        sweep_enabled = bool(params.get("aggregation_sweep", False)) or (
            str(agg_method_in).strip().lower() in {"auto", "all", "sweep"}
        )

        default_candidates = [
            "reconstruct_warm_raw",
            "reconstruct_warm_ewma",
            "reconstruct_warm_holt",
            "reconstruct_warm_hp",
            "hp_hist_warm",
            "hp_raw_warm",
        ]
        
        raw_candidates = params.get("aggregation_candidates", default_candidates)
        candidates = list(dict.fromkeys(coerce_str_list(raw_candidates)))
        if not candidates:
            candidates = list(dict.fromkeys(coerce_str_list(default_candidates)))
        
        
        sel_metric = str(config.get("aggregation_selection_metric",
                                    params.get("aggregation_selection_metric", "SMAPE")))
        sel_metric_upper = sel_metric.upper()

        # NEW: if VAL is going through full-sequence evaluator, aggregation sweep is meaningless
        # (aggregation_method_local is ignored in fullseq path).
        fullseq_like_val = (latent_mode == "full_sequence") or _is_fullseq_like_for_split("val", y_val_pred)
        fullseq_like_test = (latent_mode == "full_sequence") or _is_fullseq_like_for_split("test", y_test_pred)

        if sweep_enabled and fullseq_like_val:
            logger.info(
                "evaluate_seq2seq_sweep_skip reason=fullseq_eval "
                "split=val latent_mode=%s fullseq_like_val=%s agg_method=%s candidates=%d metric=%s",
                latent_mode, str(fullseq_like_val), str(agg_method_in), int(len(candidates)), str(sel_metric_upper),
            )
            sweep_enabled = False

    
        # ------------------------------------------------------------------
        # 2) Evaluate either single method or sweep
        # ------------------------------------------------------------------
        by_filter: Dict[str, Dict[str, Any]] = {}
        best_method = agg_method_in
        res_test_best = None
        res_val_best = None

        def _score_block(gm: Dict[str, Any]) -> Tuple[float, Tuple[float, float, float]]:
            r2   = float(gm.get("R²", gm.get("R2", np.nan)))
            smap = float(gm.get("SMAPE", np.nan))
            mae  = float(gm.get("MAE", np.nan))
            primary = -r2 if sel_metric_upper == "R²" else (mae if sel_metric_upper == "MAE" else smap)
            tie = (smap, mae, -r2)
            return primary, tie

        # ------------------------------------------------------------------
        # 2) Evaluate either single method or sweep
        # ------------------------------------------------------------------
        if not sweep_enabled:
            with phase(logger, "evaluate_seq2seq"):
                res_test = _eval_seq2_for_split(
                    split_name="test",
                    y_scaled=y_test_scaled,
                    y_pred_scaled=y_test_pred,
                    aggregation_method_local=agg_method_in,
                )
                res_val = _eval_seq2_for_split(
                    split_name="val",
                    y_scaled=y_val_scaled,
                    y_pred_scaled=y_val_pred,
                    aggregation_method_local=agg_method_in,
                )
            res_test_best, res_val_best = res_test, res_val
            best_method = agg_method_in
            by_filter[agg_method_in] = {
                "val": res_val.get("global_metrics", {}),
                "test": res_test.get("global_metrics", {}),
            }

        else:
            # NEW: cache fullseq splits across sweep loop to avoid repeated identical evals
            cache: Dict[str, Dict[str, Any]] = {}

            with phase(logger, "evaluate_seq2seq_sweep", candidates=len(candidates), metric=sel_metric_upper):
                best_score = (np.inf, (np.inf, np.inf, np.inf))

                for meth in candidates:
                    try:
                        if fullseq_like_test:
                            res_test = cache.get("test_fullseq")
                            if res_test is None:
                                res_test = _eval_seq2_for_split(
                                    split_name="test",
                                    y_scaled=y_test_scaled,
                                    y_pred_scaled=y_test_pred,
                                    aggregation_method_local=meth,
                                )
                                cache["test_fullseq"] = res_test
                        else:
                            res_test = _eval_seq2_for_split(
                                split_name="test",
                                y_scaled=y_test_scaled,
                                y_pred_scaled=y_test_pred,
                                aggregation_method_local=meth,
                            )

                        if fullseq_like_val:
                            res_val = cache.get("val_fullseq")
                            if res_val is None:
                                res_val = _eval_seq2_for_split(
                                    split_name="val",
                                    y_scaled=y_val_scaled,
                                    y_pred_scaled=y_val_pred,
                                    aggregation_method_local=meth,
                                )
                                cache["val_fullseq"] = res_val
                        else:
                            res_val = _eval_seq2_for_split(
                                split_name="val",
                                y_scaled=y_val_scaled,
                                y_pred_scaled=y_val_pred,
                                aggregation_method_local=meth,
                            )

                    except Exception:
                        logger.exception("aggregation_sweep_failed method=%s", meth)
                        continue

                    gm_val = res_val.get("global_metrics", {})
                    score = _score_block(gm_val)

                    by_filter[meth] = {
                        "val": gm_val,
                        "test": res_test.get("global_metrics", {}),
                    }

                    if score < best_score:
                        best_score = score
                        best_method = meth
                        res_test_best, res_val_best = res_test, res_val

            params["aggregation_method"] = best_method
            params["aggregation_sweep"] = True
            params["aggregation_selection_metric"] = sel_metric_upper
            params["aggregation_explored"] = candidates

    
        # ------------------------------------------------------------------
        # 2.5) Align aggregated series lengths for full_sequence mode only
        # (Do NOT touch OFF / legacy, to avoid surprises.)
        # ------------------------------------------------------------------
        if latent_mode == "full_sequence":
            def _align_agg(res: Optional[Dict[str, Any]], split_name: str) -> None:
                if res is None:
                    return
                agg_true = res.get("agg_y_test")
                agg_pred = res.get("agg_y_pred")
                if agg_true is None or agg_pred is None:
                    return
                agg_true = np.asarray(agg_true, dtype=float).reshape(-1)
                agg_pred = np.asarray(agg_pred, dtype=float).reshape(-1)
                if agg_true.shape[0] != agg_pred.shape[0]:
                    trim_len = min(agg_true.shape[0], agg_pred.shape[0])
                    logger.warning(
                        "latent_ctx(seq2eval) align_agg split=%s true_len=%d pred_len=%d -> trimming to %d",
                        split_name, agg_true.shape[0], agg_pred.shape[0], trim_len,
                    )
                    res["agg_y_test"] = agg_true[:trim_len]
                    res["agg_y_pred"] = agg_pred[:trim_len]
            _align_agg(res_test_best, "test")
            _align_agg(res_val_best,  "val")
    
        # ------------------------------------------------------------------
        # 3) Tag DFs with chosen method (do not mutate base_tags in place)
        # ------------------------------------------------------------------
        policy_tag = {"aggregation_method": best_method}
        base_tags_local = {**base_tags, **policy_tag}
    
        # ------------------------------------------------------------------
        # 4) Cumulative evaluation (chosen method only)
        # ------------------------------------------------------------------
        with phase(logger, "evaluate_cumulative_seq"):
            cum_test = evaluate_cumulative_seq(
                res_test_best["agg_y_test"], res_test_best["agg_y_pred"], y_train_original,
                scaler_target,
                params["lag_window"], params["horizon"],
                config=config, set_name="Cumulative", plot=False
            )
            cum_val = evaluate_cumulative_seq(
                res_val_best["agg_y_test"], res_val_best["agg_y_pred"], y_train_original,
                scaler_target,
                params["lag_window"], params["horizon"],
                config=config, set_name="Cumulative", plot=False
            )
    
        # ------------------------------------------------------------------
        # 4.7) NEW (visual-only): inject integrated-view members BEFORE plotting
        # Prefer upstream (offline_analytic coupling_spaghetti). Keep OFF intact.
        # ------------------------------------------------------------------
        try:
            # OFF: só usa members se vierem explicitamente do upstream (meta/params),
            # nunca tenta "fabricar" a partir de ribbons.
            prefer_meta_only = (latent_mode == "off")
        
            max_members = int(params.get("plot_max_members", 150))
        
            def _first_2d(*cands: Any) -> Optional[np.ndarray]:
                for c in cands:
                    a = _to_2d(c)
                    if a is not None and a.size > 0:
                        return a
                return None
        
            def _get_meta_key(meta_dict: Dict[str, Any], *keys: str) -> Any:
                for k in keys:
                    if k in meta_dict and meta_dict.get(k) is not None:
                        return meta_dict.get(k)
                return None
        
            def _get_attr_any(obj: Any, *keys: str) -> Any:
                for k in keys:
                    v = _get_attr_or_key(obj, k, None)
                    if v is not None:
                        return v
                return None
        
            # target lengths (aligned to what integrated plot expects)
            target_len_val  = int(len(np.asarray(res_val_best["agg_y_pred"]).reshape(-1)))
            target_len_test = int(len(np.asarray(res_test_best["agg_y_pred"]).reshape(-1)))
        
            # ----------------------------
            # VAL: find members (scaled or phys)
            # ----------------------------
            mv_phys = _to_2d(params.get("integrated_view_val_members"))
            mv_scaled = _first_2d(
                params.get("integrated_view_val_members_scaled"),
                _get_meta_key(meta,
                    # preferred names
                    "integrated_view_val_members_scaled",
                    "val_members_scaled",
                    "pred_val_members_scaled",
                    "coupling_val_members_scaled",
                    "coupling_spaghetti_val_members_scaled",
                ),
                _get_attr_any(ensemble_out,
                    # attr fallbacks (if you stored as attrs in EnsembleOutput)
                    "integrated_view_val_members_scaled",
                    "pred_val_members_scaled",
                    "coupling_val_members_scaled",
                    "coupling_spaghetti_val_members_scaled",
                ),
            )
        
            if mv_phys is None and mv_scaled is not None:
                # mv_scaled = mv_scaled[:max_members, :]
                mv_phys = _inverse_members_scaled(mv_scaled, scaler_target)
        
            if mv_phys is None and (not prefer_meta_only):
                # last resort: only if it is fullseq-like and you actually have K>1 windows
                if (latent_mode == "full_sequence") or _is_fullseq_like_for_split("val", y_val_pred):
                    mv_scaled_fb = _to_2d(_select_windows_for_plot_fallback("val", y_val_pred))
                    if mv_scaled_fb is not None:
                        mv_scaled_fb = mv_scaled_fb[:max_members, :]
                        mv_phys = _inverse_members_scaled(mv_scaled_fb, scaler_target)
        
            mv_phys = _crop_members_to_len(mv_phys, target_len_val) if mv_phys is not None else None
            if mv_phys is not None and mv_phys.shape[0] > 1 and "integrated_view_val_members" not in params:
                params["integrated_view_val_members"] = mv_phys
        
            # ----------------------------
            # TEST: find members (scaled or phys)  <-- o seu preset quer este
            # ----------------------------
            mt_phys = _to_2d(params.get("integrated_view_test_members"))
            mt_scaled = _first_2d(
                params.get("integrated_view_test_members_scaled"),
                _get_meta_key(meta,
                    "integrated_view_test_members_scaled",    # preferred
                    "test_members_scaled",
                    "pred_test_members_scaled",
                    "coupling_test_members_scaled",
                    "coupling_spaghetti_test_members_scaled",
                    # às vezes vem como "members" sem sufixo
                    "integrated_view_test_members",
                    "test_members",
                ),
                _get_attr_any(ensemble_out,
                    "integrated_view_test_members_scaled",
                    "pred_test_members_scaled",
                    "coupling_test_members_scaled",
                    "coupling_spaghetti_test_members_scaled",
                ),
            )
        
            if mt_phys is None and mt_scaled is not None:
                mt_phys = _inverse_members_scaled(mt_scaled, scaler_target)

        
            if mt_phys is None and (not prefer_meta_only):
                if (latent_mode == "full_sequence") or _is_fullseq_like_for_split("test", y_test_pred):
                    mt_scaled_fb = _to_2d(_select_windows_for_plot_fallback("test", y_test_pred))
                    if mt_scaled_fb is not None:
                        mt_scaled_fb = mt_scaled_fb[:max_members, :]
                        mt_phys = _inverse_members_scaled(mt_scaled_fb, scaler_target)
        
            mt_phys = _crop_members_to_len(mt_phys, target_len_test) if mt_phys is not None else None
            if mt_phys is not None and mt_phys.shape[0] > 1 and "integrated_view_test_members" not in params:
                params["integrated_view_test_members"] = mt_phys
        
            logger.info(
                "integrated_view_members_ready val=%s test=%s (prefer_meta_only=%s)",
                None if params.get("integrated_view_val_members") is None else np.asarray(params["integrated_view_val_members"]).shape,
                None if params.get("integrated_view_test_members") is None else np.asarray(params["integrated_view_test_members"]).shape,
                str(prefer_meta_only),
            )
        
        except Exception:
            logger.debug("integrated_view_members injection failed", exc_info=True)

    
        # ------------------------------------------------------------------
        # 5) Plots (unchanged, but now integrated plot sees members in params)
        # ------------------------------------------------------------------
        with phase(logger, "plot_seq_views"):
            _plot_seq(truth=res_test_best["agg_y_test"], pred=res_test_best["agg_y_pred"],
                      metrics=res_test_best["global_metrics"], label=label,
                      scaler_target=scaler_target, params=params, well=well,
                      is_cum=False, split="test", plot=plot, ensemble_out=ensemble_out)
            _plot_seq(truth=cum_test["y_test_cumsum"], pred=cum_test["y_pred_cumsum"],
                      metrics=cum_test["global_metrics"], label=label,
                      scaler_target=scaler_target, params=params, well=well,
                      is_cum=True, split="test", plot=plot, ensemble_out=ensemble_out)
            _plot_seq(truth=res_val_best["agg_y_test"], pred=res_val_best["agg_y_pred"],
                      metrics=res_val_best["global_metrics"], label=label,
                      scaler_target=scaler_target, params=params, well=well,
                      is_cum=False, split="val", plot=plot, ensemble_out=ensemble_out)
            _plot_seq(truth=cum_val["y_test_cumsum"], pred=cum_val["y_pred_cumsum"],
                      metrics=cum_val["global_metrics"], label=label,
                      scaler_target=scaler_target, params=params, well=well,
                      is_cum=True, split="val", plot=plot, ensemble_out=ensemble_out)
    
        with phase(logger, "plot_integrated_view"):
            _plot_integrated_view_from_agg(
                agg_val_true=res_val_best["agg_y_test"], agg_val_pred=res_val_best["agg_y_pred"],
                agg_test_true=res_test_best["agg_y_test"], agg_test_pred=res_test_best["agg_y_pred"],
                y_train_original=y_train_original,
                x_train_windows=x_train_main_windows,
                params=params, label=label, well=well, config=config,
                metrics_val_agg=res_val_best["global_metrics"],
                metrics_test_agg=res_test_best["global_metrics"],
                metrics_val_cum=cum_val["global_metrics"],
                metrics_test_cum=cum_test["global_metrics"],
                scaler_target=scaler_target,
                plot=plot
            )
    
        # ------------------------------------------------------------------
        # 6) Pack results (with base_tags_local)
        # ------------------------------------------------------------------
        agg_test_df = pd.DataFrame([_metrics_row(
            metrics=res_test_best["global_metrics"], well=well, method=arch,
            category="Aggregated", kind="Aggregated_Window", extra_tags=base_tags_local
        )])
        cum_test_df = pd.DataFrame([_metrics_row(
            metrics=cum_test["global_metrics"], well=well, method=arch,
            category="Cumulative", kind="Cumulative_Sum", extra_tags=base_tags_local
        )])
        gm_test = {**base_tags_local, **res_test_best["global_metrics"], "Category": "Global", "Kind": "Overall"}
    
        agg_val_df = pd.DataFrame([_metrics_row(
            metrics=res_val_best["global_metrics"], well=well, method=arch,
            category="Aggregated", kind="Aggregated_Window", extra_tags=base_tags_local
        )])
        cum_val_df = pd.DataFrame([_metrics_row(
            metrics=cum_val["global_metrics"], well=well, method=arch,
            category="Cumulative", kind="Cumulative_Sum", extra_tags=base_tags_local
        )])
        gm_val = {**base_tags_local, **res_val_best["global_metrics"], "Category": "Global", "Kind": "Overall"}
    
        # ------------------------------------------------------------------
        # 7) series_artifacts for Series Store / self-heal (+ optional members)
        # ------------------------------------------------------------------
        val_len = len(res_val_best["agg_y_test"])
        test_len = len(res_test_best["agg_y_test"])
        val_t  = _extract_split_timestamps(scaler_target, "val",  val_len)
        test_t = _extract_split_timestamps(scaler_target, "test", test_len)
    
        series_artifacts = {
            "val": {
                "t": val_t,
                "yhat": _safe_to_list(res_val_best["agg_y_pred"]),
                "ytrue": _safe_to_list(res_val_best["agg_y_test"]),
            },
            "test": {
                "t": test_t,
                "yhat": _safe_to_list(res_test_best["agg_y_pred"]),
                "ytrue": _safe_to_list(res_test_best["agg_y_test"]),
            },
        }
    
        # Visual-only: attach members to artifacts if present
        if "integrated_view_val_members" in params:
            try:
                series_artifacts["val"]["members"] = _safe_to_list(params["integrated_view_val_members"])
            except Exception:
                pass
        if "integrated_view_test_members" in params:
            try:
                series_artifacts["test"]["members"] = _safe_to_list(params["integrated_view_test_members"])
            except Exception:
                pass
    
        # ------------------------------------------------------------------
        # 8) Optional: record sweep diagnostics for later inspection
        # ------------------------------------------------------------------
        params.setdefault("_aggregation_sweep_info", {})
        params["_aggregation_sweep_info"].update({
            "chosen_filter": best_method,
            "selection_metric": sel_metric_upper,
            "explored_filters": candidates if sweep_enabled else [best_method],
            "by_filter": by_filter,
        })
    
        return (
            agg_test_df, cum_test_df, gm_test,
            agg_val_df,  cum_val_df,  gm_val,
            series_artifacts,
        )



    def _point_branch() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any],
                                 pd.DataFrame, pd.DataFrame, Dict[str, Any],
                                 Dict[str, Any]]:
        """
        Point-forecast evaluation (inverse-transform → metrics → plots).
        Returns:
          (agg_test_df, cum_test_df, gm_test, agg_val_df, cum_val_df, gm_val, series_artifacts)
    
        Plug-and-play enhancement (non-breaking):
          - If ARPS ensemble members exist (either in params["integrated_view_*_members"]
            or in ensemble_out meta as *_members_scaled), attach them to series_artifacts
            so Series Store / self-heal / plotting can reuse the same contract.
        """
        from evaluation.evaluation import calculate_rolling_ape
    
        # 1) Inverse-transform to physical domain
        with phase(logger, "evaluate_point_forecast"):
            y_test_true      = _inverse_transform_1d(scaler_target, y_test_scaled)
            y_test_pred_inv  = _inverse_transform_1d(scaler_target, y_test_pred)
            y_val_true       = _inverse_transform_1d(scaler_target, y_val_scaled)
            y_val_pred_inv   = _inverse_transform_1d(scaler_target, y_val_pred)
    
            # Build cumulative series anchored at end of training cumulative
            train_series = _reconstruct_train_series_phys(y_train_original, scaler_target)
            train_cum    = np.cumsum(train_series)
            anchor       = float(train_cum[-1]) if train_cum.size > 0 else 0.0
    
            test_cum_true = np.cumsum(y_test_true) + anchor
            test_cum_pred = np.cumsum(y_test_pred_inv) + anchor
            val_cum_true  = np.cumsum(y_val_true) + anchor
            val_cum_pred  = np.cumsum(y_val_pred_inv) + anchor
    
            # Metrics (aggregated & cumulative)
            met_test_agg = _compute_metrics(y_test_true, y_test_pred_inv)
            met_val_agg  = _compute_metrics(y_val_true,  y_val_pred_inv)
            met_test_cum = _compute_metrics(test_cum_true, test_cum_pred)
            met_val_cum  = _compute_metrics(val_cum_true,  val_cum_pred)
    
            # Optional: rolling APE-based cumulative metric override
            mode = params.get("cum_metric_mode", config.get("cum_metric_mode", "rolling_ape"))
            if mode == "rolling_ape":
                step = int(params.get("cum_metric_step", 15))
    
                horizon_test = int(params.get("horizon", len(y_test_true)))
                rolling_test_results = calculate_rolling_ape(
                    y_true_rates=y_test_true, y_pred_rates=y_test_pred_inv,
                    horizon=horizon_test, step=step
                )
                if rolling_test_results:
                    met_test_cum["SMAPE"] = rolling_test_results.get("APE_total_rolling_mean")
                    met_test_cum.update(rolling_test_results)
    
                horizon_val = int(params.get("horizon", len(y_val_true)))
                rolling_val_results = calculate_rolling_ape(
                    y_true_rates=y_val_true, y_pred_rates=y_val_pred_inv,
                    horizon=horizon_val, step=step
                )
                if rolling_val_results:
                    met_val_cum["SMAPE"] = rolling_val_results.get("APE_total_rolling_mean")
                    met_val_cum.update(rolling_val_results)
    
        # 2) Plots (unchanged)
        with phase(logger, "plot_point_forecast"):
            _plot_seq(truth=y_test_true, pred=y_test_pred_inv, metrics=met_test_agg,
                      label=label, scaler_target=scaler_target, params=params, well=well,
                      is_cum=False, split="test", plot=plot, ensemble_out=ensemble_out)
            _plot_seq(truth=test_cum_true, pred=test_cum_pred, metrics=met_test_cum,
                      label=label, scaler_target=scaler_target, params=params, well=well,
                      is_cum=True, split="test", plot=plot, ensemble_out=ensemble_out)
            _plot_seq(truth=y_val_true, pred=y_val_pred_inv, metrics=met_val_agg,
                      label=label, scaler_target=scaler_target, params=params, well=well,
                      is_cum=False, split="val", plot=plot, ensemble_out=ensemble_out)
            _plot_seq(truth=val_cum_true, pred=val_cum_pred, metrics=met_val_cum,
                      label=label, scaler_target=scaler_target, params=params, well=well,
                      is_cum=True, split="val", plot=plot, ensemble_out=ensemble_out)
    
            # Optional integrated view for ARPS
            is_arps_local = isinstance(arch, str) and arch.startswith("Arps_")
            if is_arps_local and plot and (x_train_main_windows is not None):
                with phase(logger, "evaluate_arps_point_and_cumulative"):
                    try:
                        from forecast_pipeline.plotting import (
                            plot_arps_integrated_from_point,
                            plot_integrated_view as _plot_integrated_view_public,
                        )
                        # ------------------------------------------------------------------
                        # NEW (ARPS-only, visual): inject spaghetti members into params
                        # so plot_arps_integrated_from_point / integrated views can reuse it.
                        # Source: ensemble_out["meta"]["integrated_view_{split}_members_scaled"]
                        # ------------------------------------------------------------------
                        try:
                            if isinstance(ensemble_out, dict):
                                meta = ensemble_out.get("meta", {})
                                if not isinstance(meta, dict):
                                    meta = {}
                        
                                def _to_2d(a: Any) -> Optional[np.ndarray]:
                                    if a is None:
                                        return None
                                    arr = np.asarray(a, dtype=float)
                                    if arr.size == 0:
                                        return None
                                    if arr.ndim == 1:
                                        arr = arr.reshape(1, -1)
                                    if arr.ndim != 2:
                                        return None
                                    return arr
                        
                                def _inv_members_scaled(members_scaled_2d: Any) -> Optional[np.ndarray]:
                                    m2d = _to_2d(members_scaled_2d)
                                    if m2d is None:
                                        return None
                                    K, H = m2d.shape
                                    try:
                                        nfi = getattr(scaler_target, "n_features_in_", None)
                                        if nfi == 1:
                                            flat = m2d.reshape(-1, 1)
                                            inv = scaler_target.inverse_transform(flat).reshape(K, H)
                                            return inv
                                        if isinstance(nfi, (int, np.integer)) and int(nfi) == int(H):
                                            return scaler_target.inverse_transform(m2d)
                                        # fallback: treat as (K*H,1)
                                        flat = m2d.reshape(-1, 1)
                                        inv = scaler_target.inverse_transform(flat).reshape(K, H)
                                        return inv
                                    except Exception:
                                        logger.debug("ARPS integrated members: inverse_transform failed; keeping scaled.")
                                        return m2d
                        
                                # VAL members
                                if "integrated_view_val_members" not in params:
                                    mv_scaled = meta.get("integrated_view_val_members_scaled")
                                    mv_phys = _inv_members_scaled(mv_scaled)
                                    if mv_phys is not None and mv_phys.shape[0] > 1:
                                        params["integrated_view_val_members"] = mv_phys
                        
                                # TEST members
                                if "integrated_view_test_members" not in params:
                                    mt_scaled = meta.get("integrated_view_test_members_scaled")
                                    mt_phys = _inv_members_scaled(mt_scaled)
                                    if mt_phys is not None and mt_phys.shape[0] > 1:
                                        params["integrated_view_test_members"] = mt_phys
                        
                                logger.info(
                                    "ARPS integrated members injected val=%s test=%s",
                                    None if params.get("integrated_view_val_members") is None else np.asarray(params["integrated_view_val_members"]).shape,
                                    None if params.get("integrated_view_test_members") is None else np.asarray(params["integrated_view_test_members"]).shape,
                                )
                        except Exception:
                            logger.debug("ARPS integrated members injection failed", exc_info=True)

                        plot_arps_integrated_from_point(
                            agg_val_true=y_val_true, agg_val_pred=y_val_pred_inv,
                            agg_test_true=y_test_true, agg_test_pred=y_test_pred_inv,
                            y_train_windows=y_train_original, x_train_windows=x_train_main_windows,
                            scaler_target=scaler_target, params=params, label=label, well=well,
                            metrics_val_agg=met_val_agg, metrics_test_agg=met_test_agg,
                            metrics_val_cum=met_val_cum, metrics_test_cum=met_test_cum,
                            plot_integrated_view_fn=_plot_integrated_view_public,
                        )
                    except Exception:
                        logger.exception("integrated_plot_arps_failed")
    
        # 3) Pack results (original schema)
        agg_test_df = pd.DataFrame([_metrics_row(
            metrics=met_test_agg, well=well, method=arch,
            category="Aggregated", kind="Series", extra_tags=base_tags
        )])
        cum_test_df = pd.DataFrame([_metrics_row(
            metrics=met_test_cum, well=well, method=arch,
            category="Cumulative", kind="Series", extra_tags=base_tags
        )])
        gm_test = {**base_tags, **met_test_agg, "Category": "Global", "Kind": "Overall"}
    
        agg_val_df = pd.DataFrame([_metrics_row(
            metrics=met_val_agg, well=well, method=arch,
            category="Aggregated", kind="Series", extra_tags=base_tags
        )])
        cum_val_df = pd.DataFrame([_metrics_row(
            metrics=met_val_cum, well=well, method=arch,
            category="Cumulative", kind="Series", extra_tags=base_tags
        )])
        gm_val = {**base_tags, **met_val_agg, "Category": "Global", "Kind": "Overall"}
    
        # 4) series_artifacts for Series Store / self-heal (+ optional members)
        val_t  = _extract_split_timestamps(scaler_target, "val",  len(y_val_true))
        test_t = _extract_split_timestamps(scaler_target, "test", len(y_test_true))
    
        series_artifacts = {
            "val": {
                "t": val_t,
                "yhat": _safe_to_list(y_val_pred_inv),
                "ytrue": _safe_to_list(y_val_true),
            },
            "test": {
                "t": test_t,
                "yhat": _safe_to_list(y_test_pred_inv),
                "ytrue": _safe_to_list(y_test_true),
            },
        }
    
        # ------------------------------------------------------------------
        # 4.1) NEW (visual-only): attach ensemble members if available
        # ------------------------------------------------------------------
        try:
            # (A) Preferred: already in physical units (injected upstream)
            mv_phys = params.get("integrated_view_val_members")
            mt_phys = params.get("integrated_view_test_members")
    
            if mv_phys is not None:
                series_artifacts["val"]["members"] = _safe_to_list(np.asarray(mv_phys, dtype=float))
            if mt_phys is not None:
                series_artifacts["test"]["members"] = _safe_to_list(np.asarray(mt_phys, dtype=float))
    
            # (B) Fallback: scaled members stored in ensemble_out["meta"], invert to phys
            if isinstance(ensemble_out, dict):
                meta = ensemble_out.get("meta", {}) if isinstance(ensemble_out.get("meta", {}), dict) else {}
    
                def _inv_scaled_members(m2d_scaled: Any) -> Optional[np.ndarray]:
                    if m2d_scaled is None:
                        return None
                    m2d = np.asarray(m2d_scaled, dtype=float)
                    if m2d.ndim == 1:
                        m2d = m2d.reshape(1, -1)
                    if m2d.ndim != 2 or m2d.size == 0:
                        return None
                    K, H = m2d.shape
                    flat = m2d.reshape(-1, 1)
                    inv = scaler_target.inverse_transform(flat).reshape(K, H)
                    return inv
    
                if "members" not in series_artifacts["val"]:
                    mv_scaled = meta.get("integrated_view_val_members_scaled")
                    mv_inv = _inv_scaled_members(mv_scaled)
                    if mv_inv is not None:
                        series_artifacts["val"]["members"] = _safe_to_list(mv_inv)
    
                if "members" not in series_artifacts["test"]:
                    mt_scaled = meta.get("integrated_view_test_members_scaled")
                    mt_inv = _inv_scaled_members(mt_scaled)
                    if mt_inv is not None:
                        series_artifacts["test"]["members"] = _safe_to_list(mt_inv)
    
        except Exception:
            logger.debug("point_branch: attaching members failed", exc_info=True)
    
        return (
            agg_test_df, cum_test_df, gm_test,
            agg_val_df,  cum_val_df,  gm_val,
            series_artifacts,
        )

    
    # ------- FINAL DISPATCH / RETURN (MISSING BEFORE) -------
    logger = get_logger(__name__)
    arch = str(params.get("architecture_name", "") or "")

    # tags/label for rows & plots
    base_tags, label, _is_darts, _darts_model = build_tags_and_label(params, well)

    # Decide the evaluation path
    is_seq2, reason = _is_seq2seq_path(arch, y_test_scaled)
    _maybe_log(config, f"[evaluate_job] path_decision={('seq2seq' if is_seq2 else 'point')} reason={reason}")

    if is_seq2:
        # Seq2* evaluation: returns 7 items (incl. series_artifacts)
        return _seq2seq_branch()
    else:
        # Optional ARPS extras (plots + cumulative) before point branch packing
        _arps_prebranch_if_needed(arch)
        # Point-forecast evaluation: returns 7 items (incl. series_artifacts)
        return _point_branch()




def evaluate_cumulative_arps(
    y_true_phys: np.ndarray,              # rates in physical units
    y_pred_phys: np.ndarray,              # rates in physical units
    y_train_windows: np.ndarray,
    x_train_main_windows: Optional[np.ndarray],
    scaler_target,
    config: Dict[str, Any],
    set_name: str,
    params: Dict[str, Any],
    plot: bool,
) -> Dict[str, Any]:
    """
    ARPS-only cumulative evaluation.
    The "SMAPE" metric is replaced by Rolling APE_total on the underlying rates by default.
    """
    # --- NEW: Import the rolling APE calculator ---
    from evaluation.evaluation import calculate_rolling_ape

    train_rest_phys = _reconstruct_train_series_phys(y_train_windows, scaler_target)

    if x_train_main_windows is not None and getattr(x_train_main_windows, "size", 0) > 0:
        tgt0_scaled = x_train_main_windows[0, :, -1].reshape(-1, 1)
        prefix_phys = scaler_target.inverse_transform(tgt0_scaled).reshape(-1)
        train_with_prefix = np.concatenate([prefix_phys, train_rest_phys], axis=0)
    else:
        train_with_prefix = train_rest_phys

    anchor = float(np.cumsum(train_with_prefix)[-1]) if train_with_prefix.size > 0 else 0.0

    y_true_cum = np.cumsum(y_true_phys) + anchor
    y_pred_cum = np.cumsum(y_pred_phys) + anchor
    
    # --- NEW: Replace metric calculation logic ---
    met_cum = _compute_metrics(y_true_cum, y_pred_cum)

    mode = params.get("cum_metric_mode", config.get("cum_metric_mode", "rolling_ape"))

    if mode == "rolling_ape":
        rolling_results = calculate_rolling_ape(
            y_true_rates=y_true_phys,
            y_pred_rates=y_pred_phys,
            horizon=int(params.get("horizon", len(y_true_phys))),
            step=int(params.get("cum_metric_step", 15)) # Use step=15 as default
        )
        if rolling_results:
            met_cum["SMAPE"] = rolling_results.get('APE_total_rolling_mean')
            met_cum.update(rolling_results)

    if plot:
        from forecast_pipeline.plotting import _plot_seq
        label = f"{config['wells'][0]} │ {params.get('architecture_name', 'Arps')}"
        _plot_seq(
            truth=y_true_cum, pred=y_pred_cum, metrics=met_cum,
            label=label, scaler_target=scaler_target, params=params,
            well=config["wells"][0], is_cum=True, split=("val" if len(y_true_phys)==len(y_pred_phys) else "test"),
            plot=True, ensemble_out=None
        )

    return {
        "y_test_cumsum": y_true_cum,
        "y_pred_cumsum": y_pred_cum,
        "global_metrics": met_cum,
    }


# ----------------------------- small utilities -----------------------------

def _maybe_log(cfg: Dict[str, Any], *args):
    """Tiny optional logger that respects config['debug_eval']."""
    if cfg.get("debug_eval", False):
        print(*args)


def _get_true_prefix_scaled(scaler_y, split_name: str) -> Optional[np.ndarray]:
    """
    Read TRUE warm prefix (scaled) from scaler_y._split_ctx, accepting legacy aliases.
    Returns a 1D np.ndarray or None.
    """
    sidecar = getattr(scaler_y, "_split_ctx", {}) or {}
    if split_name == "val":
        arr = sidecar.get("y_val_left_true_scaled_1d", sidecar.get("y_val_left_true_scaled", None))
    else:
        arr = sidecar.get("y_test_left_true_scaled_1d", sidecar.get("y_test_left_true_scaled", None))
    if arr is None or np.size(arr) == 0:
        return None
    return np.asarray(arr, float).reshape(-1)


def _get_pred_left_scaled(ensemble_out, split_name: str) -> Optional[np.ndarray]:
    """
    Read predicted LEFT windows (scaled) from ensemble_out.meta.
    Returns a 2D np.ndarray (K, H) or None.
    """
    if ensemble_out is None:
        return None
    meta = getattr(ensemble_out, "meta", None) or {}
    key = "pred_val_left" if split_name == "val" else "pred_test_left"
    arr = meta.get(key, None)
    if arr is None or np.size(arr) == 0:
        return None
    return np.asarray(arr, float)


def _inverse_1d(scaler_y, x1d_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform a 1D scaled array to physical domain, returning 1D."""
    return scaler_y.inverse_transform(x1d_scaled.reshape(-1, 1)).reshape(-1)


def _decide_policy(p_in: str, warm_available: bool) -> str:
    """Downgrade warm policy if no warm context is available."""
    if warm_available:
        return p_in
    if p_in == "hp_hist_warm":
        return "hp_hist"
    if p_in == "hp_raw_warm":
        return "hp_raw"
    if p_in.startswith("reconstruct_warm"):
        return "reconstruct"
    return p_in  # already non-warm


def _aggregate_with_warm(
    y_pred_inv_full: np.ndarray,
    policy: str,
    hp_lambda: float,
    *,
    warm_true_prefix_phys: Optional[np.ndarray],
    warm_pred_left_phys: Optional[np.ndarray],
    warm_kinds: set,
    fallback_logger=lambda *a, **k: None,
):
    """
    Call aggregate_predictions with available warm contexts.
    Handles older signatures gracefully (TypeError/ValueError).
    """

    try:
        kwargs = {"policy": policy, "hp_lambda": hp_lambda}
        if warm_true_prefix_phys is not None and warm_true_prefix_phys.size > 0:
            kwargs["warm_true_prefix_1d"] = warm_true_prefix_phys
        if warm_pred_left_phys is not None and warm_pred_left_phys.size > 0:
            kwargs["warm_left_windows_2d"] = warm_pred_left_phys
        return aggregate_predictions(y_pred_inv_full, **kwargs)

    except TypeError:
        # Older API without warm_true_prefix_1d
        fallback_logger("aggregate_predictions: TRUE prefix arg unsupported; falling back.")
        if (warm_pred_left_phys is not None and warm_pred_left_phys.size > 0 and policy in warm_kinds):
            return aggregate_predictions(
                y_pred_inv_full, policy=policy, hp_lambda=hp_lambda,
                warm_left_windows_2d=warm_pred_left_phys
            )
        base = "hp_hist" if policy.startswith("hp_hist") else ("hp_raw" if policy.startswith("hp_raw") else "reconstruct")
        return aggregate_predictions(y_pred_inv_full, policy=base, hp_lambda=hp_lambda)

    except ValueError as ve:
        # Some warm policies may *require* warm_left_windows_2d; fallback to non-warm sibling
        msg = str(ve).lower()
        if "requires warm_left_windows_2d" in msg or "requer warm_left_windows_2d" in msg:
            fallback = "hp_hist" if policy == "hp_hist_warm" else "hp_raw"
            fallback_logger(f"aggregate_predictions: fallback to '{fallback}' (warm required).")
            return aggregate_predictions(y_pred_inv_full, policy=fallback, hp_lambda=hp_lambda)
        raise


def _corr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of the overlapping head; NaN if constant or empty."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    m = min(a.size, b.size)
    if m == 0:
        return float("nan")
    a, b = a[:m], b[:m]
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _apply_alignment(
    agg_y_test: np.ndarray,
    agg_y_pred: np.ndarray,
    warm_true_prefix_phys: Optional[np.ndarray],
    output_length: int,
    mode: str,
    *,
    debug_log=lambda *a: None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply size-conscious alignment prior to metrics.
    Modes:
      - "drop_warm": if len(pred) == len(true) + (H-1), drop first (H-1) from predictions
      - "truth_prefixed": if prefix exists and len(true)+(H-1) == len(pred), prepend tail(prefix) to truth
      - "auto": heuristic (corr>=0.80) picks one of the above when size gap is exactly (H-1)
      - "none" or anything else: no-op
    """
    Hm1 = max(0, int(output_length) - 1)
    if Hm1 == 0:
        return agg_y_test, agg_y_pred  # nothing to align

    n_true, n_pred = int(np.size(agg_y_test)), int(np.size(agg_y_pred))
    have_prefix = warm_true_prefix_phys is not None and warm_true_prefix_phys.size > 0

    if mode == "drop_warm":
        if n_pred == n_true + Hm1:
            return agg_y_test, agg_y_pred[Hm1:]
        return agg_y_test, agg_y_pred  # sizes did not match the expected pattern

    if mode == "truth_prefixed":
        if have_prefix and (n_true + Hm1 == n_pred):
            tail = warm_true_prefix_phys[-Hm1:]
            return np.concatenate([tail, agg_y_test]), agg_y_pred
        return agg_y_test, agg_y_pred

    if mode == "auto":
        # Only act if the size gap is exactly H-1
        if n_pred == n_true + Hm1 and have_prefix:
            sim = _corr_similarity(agg_y_pred[:Hm1], warm_true_prefix_phys[-Hm1:])
            debug_log(f"auto-align corr(pred[:H-1], prefix_tail)={sim:.3f}")
            if not np.isnan(sim) and sim >= 0.80:
                return agg_y_test, agg_y_pred[Hm1:]
        elif n_true + Hm1 == n_pred and have_prefix:
            sim = _corr_similarity(agg_y_pred[:Hm1], warm_true_prefix_phys[-Hm1:])
            debug_log(f"auto-align corr(pred[:H-1], prefix_tail)={sim:.3f}")
            if not np.isnan(sim) and sim >= 0.80:
                tail = warm_true_prefix_phys[-Hm1:]
                return np.concatenate([tail, agg_y_test]), agg_y_pred
        return agg_y_test, agg_y_pred

    # "none" or unknown -> no-op
    return agg_y_test, agg_y_pred


def _equalize_lengths(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Final guard: clip both series to the same min length."""
    n_t, n_p = int(np.size(y_true)), int(np.size(y_pred))
    if n_t == n_p:
        return y_true, y_pred
    m = min(n_t, n_p)
    return y_true[:m], y_pred[:m]


def _build_full_truth_for_split_k(
    y_split_scaled: np.ndarray,
    scaler_y,
    *,
    target_length: Optional[int],
    split_name: str,
) -> np.ndarray:
    """
    Reconstructs the continuous ground-truth series (val/test) in physical space
    using the classic N+H-1 reconstruction from windowed y_split_scaled.
    Logic unchanged; logging compacted.
    """
    logger = get_logger(__name__)

    from common.log_utils import is_compact_logging, arr_fingerprint, info_v
    compact = is_compact_logging(None)

    def _qflat(x: np.ndarray, ps=(1, 5, 50, 95, 99)) -> Dict[str, float]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            return {f"p{p:02d}": float("nan") for p in ps}
        return {f"p{p:02d}": float(np.percentile(x, p)) for p in ps}

    def _inverse_2d(m2d: np.ndarray, scaler, *, tag: str) -> np.ndarray:
        m2d = np.asarray(m2d, dtype=float)
        if scaler is None:
            logger.debug("%s inverse: scaler=None -> passthrough", tag)
            return m2d

        nfi = getattr(scaler, "n_features_in_", None)
        Nn, Hh = m2d.shape

        try:
            if isinstance(nfi, (int, np.integer)) and int(nfi) == 1:
                flat = m2d.reshape(-1, 1)
                inv = scaler.inverse_transform(flat).reshape(Nn, Hh)
                logger.debug("%s inverse: path=flatten1D nfi=1 in=%s out=%s", tag, m2d.shape, inv.shape)
                return inv

            if isinstance(nfi, (int, np.integer)) and int(nfi) == int(Hh):
                inv = scaler.inverse_transform(m2d)
                logger.debug("%s inverse: path=direct2D nfi=%d in=%s out=%s", tag, int(nfi), m2d.shape, inv.shape)
                return inv

            flat = m2d.reshape(-1, 1)
            inv = scaler.inverse_transform(flat).reshape(Nn, Hh)
            logger.warning("%s inverse: path=fallback_flatten nfi=%r in=%s out=%s", tag, nfi, m2d.shape, inv.shape)
            return inv

        except Exception:
            logger.exception("%s inverse FAILED scaler=%s nfi=%r in=%s", tag, type(scaler).__name__, nfi, m2d.shape)
            return m2d

    y_split_scaled = np.asarray(y_split_scaled, dtype=float)
    if y_split_scaled.ndim != 2:
        raise ValueError(
            f"_build_full_truth_for_split_k expects 2D y_split_scaled, got shape={y_split_scaled.shape} split={split_name!r}"
        )

    # Compact: avoid multiple INFO lines. Keep details available in verbose-only logs.
    logger.debug(
        "FullSeq truth input split=%s scaled_fp=%s scaler=%s nfi=%r",
        split_name, arr_fingerprint(y_split_scaled),
        type(scaler_y).__name__ if scaler_y is not None else "None",
        getattr(scaler_y, "n_features_in_", None) if scaler_y is not None else None,
    )
    info_v(
        "fullseq_truth_input split=%s y_split_scaled_shape=%s y_split_scaled_q=%s scaler=%s nfi=%r",
        split_name, y_split_scaled.shape, _qflat(y_split_scaled),
        type(scaler_y).__name__ if scaler_y is not None else "None",
        getattr(scaler_y, "n_features_in_", None) if scaler_y is not None else None,
    )

    # Safe inverse (N,H) -> physical (N,H)
    y_split_inv = _inverse_2d(y_split_scaled, scaler_y, tag=f"fullseq_truth[{split_name}]")

    info_v("fullseq_truth_phys_windows split=%s y_split_phys_q=%s", split_name, _qflat(y_split_inv))

    # Reconstruct continuous 1D series (N+H-1)
    y_true_full = reconstruct_true_series(y_split_inv)
    full_len = int(y_true_full.size)

    info_v("fullseq_truth_reconstructed split=%s full_len=%d y_true_full_q=%s", split_name, full_len, _qflat(y_true_full))

    if target_length is not None and int(target_length) > 0:
        tgt = int(target_length)
        if tgt < full_len:
            logger.debug("FullSeq truth: trimming split=%s from %d to %d", split_name, full_len, tgt)
            y_true_full = y_true_full[:tgt]
        elif tgt > full_len:
            logger.debug("FullSeq truth: target_length=%d > full_len=%d (split=%s); keeping full_len.", tgt, full_len, split_name)

    logger.debug(
        "FullSeq truth built split=%s final_len=%d (natural=%d target=%s)",
        split_name, int(y_true_full.size), full_len, target_length,
    )
    return y_true_full



def _select_prediction_windows_fullseq(
    *,
    split_name: str,
    K: int,
    y_pred_split_scaled: np.ndarray,
    target_length: Optional[int],
) -> Tuple[np.ndarray, int]:
    """
    Selects the K prediction windows to aggregate (always the LAST K windows).
    Logic unchanged; logging compacted.
    """
    logger = get_logger(__name__)
    from common.log_utils import is_compact_logging

    compact = is_compact_logging(None)

    if K <= 0:
        raise ValueError(f"_select_prediction_windows_fullseq: K must be >=1, got {K}")

    all_pred = np.asarray(y_pred_split_scaled, dtype=float)

    if all_pred.ndim != 2:
        raise ValueError(
            f"_select_prediction_windows_fullseq expected 2D predictions, got shape={all_pred.shape} (split={split_name!r})"
        )

    n_windows, horizon = all_pred.shape
    if n_windows == 0:
        raise ValueError(f"_select_prediction_windows_fullseq: no prediction windows available for split={split_name!r}.")

    # Trim horizon if needed (same logic)
    if target_length is not None and int(target_length) > 0 and horizon > int(target_length):
        tgt = int(target_length)
        logger.debug("FullSeq windows: trimming horizon split=%s from %d to %d", split_name, horizon, tgt)
        all_pred = all_pred[:, :tgt]
        horizon = tgt

    # Effective K (same logic)
    K_eff = min(int(K), n_windows)

    # Always take the LAST K windows (same logic)
    start_idx = n_windows - K_eff
    pred_windows = all_pred[start_idx:, :]

    # Minimal debug only; caller summarizes at INFO.
    logger.debug(
        "FullSeq windows selected split=%s n_windows=%d horizon=%d K_req=%d K_eff=%d start_idx=%d",
        split_name, n_windows, horizon, K, K_eff, start_idx,
    )

    return pred_windows, K_eff



def _aggregate_fullseq_k(
    pred_windows_phys: np.ndarray,
    *,
    agg_kind: str = "mean",
) -> np.ndarray:
    """
    Agrega K ribbons de predição full-sequence em um único ribbon.

    Args:
        pred_windows_phys: 2D array (K, N_split) no domínio físico.
        agg_kind:          "mean" (por enquanto apenas média).

    Returns:
        1D np.ndarray de tamanho N_split com a predição agregada.
    """
    pred_windows_phys = np.asarray(pred_windows_phys, dtype=float)
    if pred_windows_phys.ndim != 2:
        raise ValueError(
            f"_aggregate_fullseq_k expects 2D array, got shape={pred_windows_phys.shape}"
        )

    K, N = pred_windows_phys.shape
    if K == 0:
        raise ValueError("_aggregate_fullseq_k: received zero windows.")
    if K == 1:
        # Atalho: se só tem uma janela, retorna ela.
        return pred_windows_phys[0].copy()

    agg_kind_norm = (agg_kind or "mean").strip().lower()
    if agg_kind_norm in {"mean", "avg", "average"}:
        return np.nanmean(pred_windows_phys, axis=0)

    logger = get_logger(__name__)
    logger.warning(
        "_aggregate_fullseq_k: unsupported agg_kind=%r, falling back to 'mean'.",
        agg_kind,
    )
    return np.nanmean(pred_windows_phys, axis=0)




def evaluate_fullseq_k_mode(
    *,
    mode: str,
    K: int,
    split_name: str,         # "val" | "test"
    y_split_scaled: np.ndarray,
    y_pred_split_scaled: np.ndarray,
    scaler_y,
    input_length: int,
    output_length: int,
    train_size: float,
    config: Dict[str, Any],
    eval_title: str,
    set_name: str,
    ensemble_out: Optional["EnsembleOutput"],  # kept for compat
    target_length: Optional[int],
    agg_kind: str = "mean",
    plot: bool = False,
) -> Dict[str, Any]:
    logger = get_logger(__name__)

    # --- reuse your logging façade ---
    from common.log_utils import (
        stage_banner,
        log_kv_block,
        effective_log_width,
        is_compact_logging,
        arr_fingerprint,
        info_v,
    )

    width = effective_log_width(None, fallback=100)
    compact = is_compact_logging(None)

    def _qflat(x: np.ndarray, ps=(1, 5, 50, 95, 99)) -> Dict[str, float]:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            return {f"p{p:02d}": float("nan") for p in ps}
        return {f"p{p:02d}": float(np.percentile(x, p)) for p in ps}

    def _looks_unscaled(arr_2d: np.ndarray) -> bool:
        # Heuristic: RobustScaler outputs are usually near 0; huge magnitudes suggest physical space.
        qs = _qflat(arr_2d, ps=(50, 95))
        med = abs(qs["p50"])
        p95 = abs(qs["p95"])
        return (med > 50.0) or (p95 > 200.0)

    def _inverse_2d(m2d: np.ndarray, scaler, *, tag: str) -> np.ndarray:
        """
        Safe inverse for 2D matrices (K,H) when the scaler may have been fit on 1 feature.
        Same logic as before; only logging verbosity changed.
        """
        m2d = np.asarray(m2d, dtype=float)
        if scaler is None:
            # High-signal only; keep it terse.
            logger.debug("%s inverse: scaler=None -> passthrough", tag)
            return m2d

        nfi = getattr(scaler, "n_features_in_", None)
        Kk, Hh = m2d.shape

        try:
            if isinstance(nfi, (int, np.integer)) and int(nfi) == 1:
                flat = m2d.reshape(-1, 1)
                inv = scaler.inverse_transform(flat).reshape(Kk, Hh)
                logger.debug("%s inverse: path=flatten1D nfi=1 in=%s out=%s", tag, m2d.shape, inv.shape)
                return inv

            if isinstance(nfi, (int, np.integer)) and int(nfi) == int(Hh):
                inv = scaler.inverse_transform(m2d)
                logger.debug("%s inverse: path=direct2D nfi=%d in=%s out=%s", tag, int(nfi), m2d.shape, inv.shape)
                return inv

            # Conservative fallback (same as before)
            flat = m2d.reshape(-1, 1)
            inv = scaler.inverse_transform(flat).reshape(Kk, Hh)
            logger.warning("%s inverse: path=fallback_flatten nfi=%r in=%s out=%s", tag, nfi, m2d.shape, inv.shape)
            return inv

        except Exception:
            logger.exception("%s inverse FAILED scaler=%s nfi=%r in=%s", tag, type(scaler).__name__, nfi, m2d.shape)
            return m2d

    # ------------------------------------------------------------------
    # Normalize mode (same logic, English messages)
    # ------------------------------------------------------------------
    mode_norm = (mode or "deploy_split_k").strip().lower()
    if mode_norm not in {"deploy_split_k", "deploy_left_k"}:
        logger.warning("evaluate_fullseq_k_mode: unknown mode=%r; falling back to 'deploy_split_k'.", mode)
        mode_norm = "deploy_split_k"

    # ------------------------------------------------------------------
    # Select prediction source: LEFT vs SPLIT (same logic)
    # ------------------------------------------------------------------
    pred_source_scaled = np.asarray(y_pred_split_scaled, dtype=float)
    source_kind = "split"

    if mode_norm == "deploy_left_k":
        left_scaled = None
        if ensemble_out is not None and getattr(ensemble_out, "meta", None):
            try:
                left_scaled = _get_pred_left_scaled(ensemble_out, split_name)
            except Exception:
                logger.exception("evaluate_fullseq_k_mode: failed to read LEFT predictions; falling back to split.")

        if left_scaled is not None and np.size(left_scaled) > 0:
            pred_source_scaled = np.asarray(left_scaled, dtype=float)
            source_kind = "left"
            logger.info("FullSeq: using LEFT predictions split=%s shape=%s", split_name, pred_source_scaled.shape)
        else:
            logger.warning("FullSeq: mode='deploy_left_k' but LEFT is missing for split=%s; using split predictions.", split_name)

    # ------------------------------------------------------------------
    # Compact banner + single entry summary (INFO)
    # ------------------------------------------------------------------
    stage_banner("EVAL", "fullseq_k_mode", f"split={split_name} mode={mode_norm} K={K} source={source_kind}", width=width)

    y_split_scaled_arr = np.asarray(y_split_scaled, dtype=float)
    entry_kv = {
        "split": split_name,
        "mode": mode_norm,
        "source": source_kind,
        "K_req": int(K),
        "agg_kind": str(agg_kind),
        "target_length": int(target_length) if (target_length is not None) else None,
        "y_true_scaled": arr_fingerprint(y_split_scaled_arr),
        "y_pred_scaled": arr_fingerprint(np.asarray(pred_source_scaled, dtype=float)),
        "scaler": type(scaler_y).__name__ if scaler_y is not None else "None",
        "scaler_nfi": getattr(scaler_y, "n_features_in_", None) if scaler_y is not None else None,
        "suspect_true_unscaled": bool(_looks_unscaled(y_split_scaled_arr)),
        "suspect_pred_unscaled": bool(_looks_unscaled(pred_source_scaled)),
    }
    log_kv_block("FullSeq K Eval — Enter", entry_kv, width=width)

    # Verbose-only deep diagnostics (kept, but not in compact mode)
    info_v(
        "fullseq_inputs split=%s mode=%s source=%s y_true_scaled_shape=%s y_pred_scaled_shape=%s "
        "y_true_scaled_q=%s y_pred_scaled_q=%s scaler=%s nfi=%r",
        split_name, mode_norm, source_kind,
        y_split_scaled_arr.shape, np.asarray(pred_source_scaled).shape,
        _qflat(y_split_scaled_arr), _qflat(pred_source_scaled),
        type(scaler_y).__name__ if scaler_y is not None else "None",
        getattr(scaler_y, "n_features_in_", None) if scaler_y is not None else None,
    )

    with phase(logger, "evaluate_fullseq_k_mode", split=split_name, mode=mode_norm, K=K, source=source_kind):

        # 1) Build continuous truth series (physical)
        y_true_full = _build_full_truth_for_split_k(
            y_split_scaled=y_split_scaled,
            scaler_y=scaler_y,
            target_length=target_length,
            split_name=split_name,
        )

        # 2) Select K prediction ribbons (scaled)
        pred_windows_scaled, K_eff = _select_prediction_windows_fullseq(
            split_name=split_name,
            K=K,
            y_pred_split_scaled=pred_source_scaled,
            target_length=target_length,
        )

        pred_windows_scaled = np.asarray(pred_windows_scaled, dtype=float)
        if pred_windows_scaled.ndim != 2:
            raise ValueError(f"evaluate_fullseq_k_mode: expected 2D pred_windows_scaled, got {pred_windows_scaled.shape}")

        # 3) Inverse predictions to physical space (safe inverse)
        pred_windows_phys = _inverse_2d(pred_windows_scaled, scaler_y, tag="fullseq_pred")

        # 4) Aggregate across K -> single series
        y_pred_full = _aggregate_fullseq_k(pred_windows_phys, agg_kind=agg_kind)

        # 5) Defensive alignment
        y_true_full = np.asarray(y_true_full, dtype=float).reshape(-1)
        y_pred_full = np.asarray(y_pred_full, dtype=float).reshape(-1)

        eval_len = int(min(y_true_full.size, y_pred_full.size))
        if eval_len <= 0:
            raise ValueError(
                f"evaluate_fullseq_k_mode split={split_name!r} produced zero-length series "
                f"(true_len={y_true_full.size}, pred_len={y_pred_full.size})."
            )

        y_true_eval = y_true_full[:eval_len]
        y_pred_eval = y_pred_full[:eval_len]

        # 6) Bias/offset diagnostics (same math)
        err = y_pred_eval - y_true_eval
        mean_err = float(np.mean(err))
        std_err  = float(np.std(err))
        mae_err  = float(np.mean(np.abs(err)))

        try:
            b_fit, a_fit = np.polyfit(y_pred_eval, y_true_eval, 1)  # returns (slope, intercept) but we keep naming consistent with your old line
            # Your old log was "lin_fit: a=<intercept> b=<slope>".
            a, b = float(a_fit), float(b_fit)
        except Exception:
            a, b = np.nan, np.nan

        # 7) Global metrics (same call)
        r2, smape, mae = evaluate(y_true_eval, y_pred_eval)

        global_metrics: Dict[str, Any] = {
            "R²": r2,
            "SMAPE": smape,
            "MAE": mae,
            "Category": "Global",
            "EvalMode": mode_norm,
            "K": int(K_eff),
            "SourceKind": source_kind,
            "BiasMean": mean_err,
            "BiasMAE": mae_err,
            "LinFit_a": a,
            "LinFit_b": b,
        }

        # ------------------------------------------------------------------
        # Compact exit summary (INFO): single block, high signal
        # ------------------------------------------------------------------
        exit_kv = {
            "K_eff": int(K_eff),
            "horizon": int(pred_windows_phys.shape[1]) if pred_windows_phys.ndim == 2 else None,
            "true_phys": arr_fingerprint(y_true_eval),
            "pred_phys": arr_fingerprint(y_pred_eval),
            "bias_mean": f"{mean_err:.3f}",
            "bias_std": f"{std_err:.3f}",
            "bias_mae": f"{mae_err:.3f}",
            "linfit_a": f"{a:.3f}" if np.isfinite(a) else "nan",
            "linfit_b": f"{b:.3f}" if np.isfinite(b) else "nan",
            "R2": f"{r2:.5f}",
            "SMAPE": f"{smape:.5f}",
            "MAE": f"{mae:.5f}",
            "eval_len": int(eval_len),
        }
        log_kv_block("FullSeq K Eval — Summary", exit_kv, width=width)

        # Verbose-only details (kept, but not in compact mode)
        info_v(
            "fullseq_true_vs_pred_range split=%s mode=%s source=%s eval_len=%d true_q=%s pred_q=%s",
            split_name, mode_norm, source_kind, eval_len, _qflat(y_true_eval), _qflat(y_pred_eval),
        )
        info_v(
            "fullseq_err_profile split=%s mode=%s source=%s err_q=%s",
            split_name, mode_norm, source_kind, _qflat(err),
        )
        info_v(
            "fullseq_bias split=%s mode=%s source=%s bias_mean=%.3f bias_std=%.3f bias_MAE=%.3f lin_fit: a=%.3f b=%.3f",
            split_name, mode_norm, source_kind, mean_err, std_err, mae_err, a, b,
        )
        info_v(
            "evaluate_fullseq_k_mode split=%s mode=%s source=%s K_req=%d K_eff=%d eval_length=%d R2=%.5f SMAPE=%.5f MAE=%.5f",
            split_name, mode_norm, source_kind, K, K_eff, eval_len, r2, smape, mae,
        )

        if plot:
            additional_params: Dict[str, Any] = {
                "window_size": int(input_length),
                "forecast_steps": int(output_length),
                "percentage_split": float(train_size),
                "eval_mode": mode_norm,
                "k_windows": int(K_eff),
                "source_kind": source_kind,
            }
            evaluate_and_plot(
                y_true=y_true_eval,
                y_pred=y_pred_eval,
                title=(eval_title + f" - FullSeq K={K_eff} [{mode_norm}/{source_kind}]").strip(),
                well=config["wells"][0],
                set_name=set_name,
                additional_params=additional_params,
            )

        return {
            "agg_y_test": y_true_eval,
            "agg_y_pred": y_pred_eval,
            "global_metrics": global_metrics,
            "eval_length": eval_len,
        }





def evaluate_model_seq(
    y_test_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler_y,
    input_length: int,
    output_length: int,
    train_size: float,
    config: Dict[str, Any],
    eval_title: str = "",
    set_name: str = "",
    aggregation_method: str = "reconstruct",  # reconstruct | hp_hist | hp_raw | hp_hist_warm | hp_raw_warm | reconstruct_warm_*
    quantiles: Optional[List[float]] = None,
    plot: bool = True,
    *,
    ensemble_out: Optional["EnsembleOutput"] = None,
    split_name: str = "val",              # "val" | "test"
    hp_lambda: float = 64000.0,
) -> Dict[str, Any]:
    """
    Non-causal seq-to-seq evaluation with optional warm-start aggregation.
    Prioritizes TRUE warm prefixes (scaler_y._split_ctx) over predicted LEFT windows,
    and applies size-conscious alignment before computing metrics.

    Returns:
        dict with:
          - "agg_y_test": 1D np.ndarray (aggregated ground-truth series, physical domain)
          - "agg_y_pred": 1D np.ndarray (aggregated prediction series, physical domain)
          - "global_metrics": {"R²": float, "SMAPE": float, "MAE": float, "Category": "Global"}
    """
    # 1) Physical domain
    y_test_inv_full = scaler_y.inverse_transform(y_test_scaled)   # (N, H)
    y_pred_inv_full = scaler_y.inverse_transform(y_pred_scaled)   # (N, H)

    # 2) Aggregate ground-truth (1D)
    agg_y_test = reconstruct_true_series(y_test_inv_full)

    # 3) Warm contexts (TRUE prefix preferred)
    true_prefix_scaled = _get_true_prefix_scaled(scaler_y, split_name)
    warm_true_prefix_phys = _inverse_1d(scaler_y, true_prefix_scaled) if true_prefix_scaled is not None else None

    pred_left_scaled = _get_pred_left_scaled(ensemble_out, split_name)
    warm_pred_left_phys = scaler_y.inverse_transform(pred_left_scaled) if pred_left_scaled is not None else None

    p_in = (aggregation_method or "reconstruct").lower()
    warm_kinds = {
        "hp_hist_warm", "hp_raw_warm",
        "reconstruct_warm_raw", "reconstruct_warm_ewma",
        "reconstruct_warm_hp", "reconstruct_warm_holt",
        "reconstruct_warm",  # legacy alias
    }
    warm_available = (warm_true_prefix_phys is not None and warm_true_prefix_phys.size > 0) or \
                     (warm_pred_left_phys is not None and warm_pred_left_phys.size > 0)
    policy = _decide_policy(p_in, warm_available)

    # 4) Aggregate predictions with (optional) warm
    agg_y_pred = _aggregate_with_warm(
        y_pred_inv_full, policy, hp_lambda,
        warm_true_prefix_phys=warm_true_prefix_phys,
        warm_pred_left_phys=warm_pred_left_phys,
        warm_kinds=warm_kinds,
        fallback_logger=lambda *a, **k: _maybe_log(config, *a),
    )

    # 5) Alignment (safe & size-conscious)
    align_mode = (config.get("warm_align_mode") or "truth_prefixed").lower()  # "auto" | "drop_warm" | "truth_prefixed" | "none"
    agg_y_test_eval, agg_y_pred_eval = _apply_alignment(
        agg_y_test, agg_y_pred, warm_true_prefix_phys, output_length, align_mode,
        debug_log=lambda *a: _maybe_log(config, *a),
    )

    # Final guarantee: equal lengths for metrics
    agg_y_test_eval, agg_y_pred_eval = _equalize_lengths(agg_y_test_eval, agg_y_pred_eval)

    # 6) Metrics
    r2, smape, mae = evaluate(agg_y_test_eval, agg_y_pred_eval)
    global_metrics = {"R²": r2, "SMAPE": smape, "MAE": mae, "Category": "Global"}

    # 7) Plot (use non-aligned series for continuity-of-view, as before)
    if plot:
        evaluate_and_plot(
            y_true=agg_y_test,
            y_pred=agg_y_pred,
            title=(eval_title + " - Aggregated").strip(),
            well=config["wells"][0],
            set_name=set_name,
            additional_params={
                "window_size": input_length,
                "forecast_steps": output_length,
                "percentage_split": train_size,
            },
        )

    return {
        "agg_y_test": agg_y_test,
        "agg_y_pred": agg_y_pred,
        "global_metrics": global_metrics,
    }



def evaluate_cumulative_seq(
    agg_y_test: np.ndarray,               # physical units (rates)
    agg_y_pred: np.ndarray,               # physical units (rates)
    y_train_original: np.ndarray,         # windows (scaled OR physical) -> auto-detect
    scaler_target,
    input_length: int,
    output_length: int,
    config: Dict[str, Any],
    set_name: str = "Cumulative",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Cumulative evaluation anchored on train cumsum.
    The "SMAPE" metric is replaced by Rolling APE_total on the underlying rates by default.

    NOTE:
      - In latent full-sequence mode the aggregated prediction series may be
        longer than the ground-truth series (because of N + H_ext - 1).
      - Metrics must be computed only on the overlapping region, i.e., up to
        min(len(y_true), len(y_pred)).
    """
    from forecast_pipeline.logging_utils import get_logger, phase
    from evaluation.evaluation import calculate_rolling_ape

    # Reuse your compact UI
    from common.log_utils import (
        stage_banner,
        log_kv_block,
        effective_log_width,
        is_compact_logging,
        arr_fingerprint,
        info_v,
    )

    logger = get_logger(__name__)
    width = effective_log_width(None, fallback=100)
    compact = is_compact_logging(None)

    mode = str(config.get("cum_metric_mode", "rolling_ape"))

    # One banner per call (high-level)
    stage_banner("EVAL", "cumulative_seq", f"set={set_name} metric_mode={mode} plot={bool(plot)}", width=width)

    with phase(logger, "evaluate_cumulative_seq"):
        # ------------------------------------------------------------------
        # 1) Reconstruct train cumsum (anchor)
        # ------------------------------------------------------------------
        train_series = _reconstruct_train_series_phys(y_train_original, scaler_target)
        y_train_cumsum = np.cumsum(train_series)
        anchor = float(y_train_cumsum[-1]) if y_train_cumsum.size > 0 else 0.0

        # ------------------------------------------------------------------
        # 2) Normalize inputs (flatten + align lengths on the RATE level)
        # ------------------------------------------------------------------
        y_true_rates = np.asarray(agg_y_test, dtype=float).reshape(-1)
        y_pred_rates = np.asarray(agg_y_pred, dtype=float).reshape(-1)

        trim_reason = None
        if y_true_rates.shape[0] != y_pred_rates.shape[0]:
            min_len = min(y_true_rates.shape[0], y_pred_rates.shape[0])
            logger.warning(
                "cum_rates_length_mismatch true_len=%d pred_len=%d -> trimming to min_len=%d",
                y_true_rates.shape[0],
                y_pred_rates.shape[0],
                min_len,
            )
            y_true_rates = y_true_rates[:min_len]
            y_pred_rates = y_pred_rates[:min_len]
            trim_reason = "rates_mismatch"

        # ------------------------------------------------------------------
        # 3) Build cumulative series with common length
        # ------------------------------------------------------------------
        y_test_cumsum = np.cumsum(y_true_rates) + anchor
        y_pred_cumsum = np.cumsum(y_pred_rates) + anchor

        # Sanity check (should always hold now, but keep for robustness)
        if y_test_cumsum.shape[0] != y_pred_cumsum.shape[0]:
            min_len = min(y_test_cumsum.shape[0], y_pred_cumsum.shape[0])
            logger.warning(
                "cum_length_mismatch_after_rates true_len=%d pred_len=%d -> trimming to min_len=%d",
                y_test_cumsum.shape[0],
                y_pred_cumsum.shape[0],
                min_len,
            )
            y_test_cumsum = y_test_cumsum[:min_len]
            y_pred_cumsum = y_pred_cumsum[:min_len]
            trim_reason = trim_reason or "cumsum_mismatch"

        # ------------------------------------------------------------------
        # 4) Base metrics on cumulative curves
        # ------------------------------------------------------------------
        global_metrics = _compute_metrics(y_test_cumsum, y_pred_cumsum)

        # ------------------------------------------------------------------
        # 5) Optional: Rolling APE on underlying rates
        # ------------------------------------------------------------------
        rolling_results = None
        if mode == "rolling_ape":
            rolling_results = calculate_rolling_ape(
                y_true_rates=y_true_rates,
                y_pred_rates=y_pred_rates,
                horizon=int(output_length),
                step=int(config.get("cum_metric_step", 15)),
            )

            # Override "SMAPE" with mean rolling APE_total, and attach richer stats
            if rolling_results:
                global_metrics["SMAPE"] = rolling_results.get("APE_total_rolling_mean")
                global_metrics.update(rolling_results)

        # ------------------------------------------------------------------
        # Compact, high-signal summary (INFO)
        # ------------------------------------------------------------------
        summary_kv = {
            "decision": "anchor",
            "anchor_last": f"{anchor:.6f}",
            "train_len": int(train_series.size),
            "eval_len": int(y_true_rates.size),
            "trim": trim_reason or "none",
            "rates_true": arr_fingerprint(y_true_rates),
            "rates_pred": arr_fingerprint(y_pred_rates),
            "cumsum_true": arr_fingerprint(y_test_cumsum),
            "cumsum_pred": arr_fingerprint(y_pred_cumsum),
            "R2": f"{global_metrics.get('R²', float('nan')):.5f}" if isinstance(global_metrics.get("R²"), (int, float, np.floating)) else global_metrics.get("R²"),
            "SMAPE": global_metrics.get("SMAPE"),
            "MAE": f"{global_metrics.get('MAE', float('nan')):.5f}" if isinstance(global_metrics.get("MAE"), (int, float, np.floating)) else global_metrics.get("MAE"),
            "metric_mode": mode,
        }
        log_kv_block("Cumulative Eval — Summary", summary_kv, width=width)

        # Verbose-only details (kept for deep debugging)
        info_v(
            "cumulative_details set=%s anchor_last=%.6f train_len=%d eval_len=%d mode=%s rolling_keys=%s",
            set_name,
            anchor,
            int(train_series.size),
            int(y_true_rates.size),
            mode,
            sorted(list((rolling_results or {}).keys())),
        )

        # ------------------------------------------------------------------
        # 6) Plot (if requested) using the aligned cumulative series
        # ------------------------------------------------------------------
        if plot:
            from evaluation.evaluation import evaluate_and_plot
            evaluate_and_plot(
                y_true=y_test_cumsum,
                y_pred=y_pred_cumsum,
                title="Cumulative Forecast",
                well=config["wells"][0],
                set_name=set_name,
                additional_params={
                    "window_size": input_length,
                    "forecast_steps": output_length,
                },
                r2=global_metrics.get("R²"),
                smape=global_metrics.get("SMAPE"),
                mae=global_metrics.get("MAE"),
            )

        return {
            "y_test_cumsum": y_test_cumsum,
            "y_pred_cumsum": y_pred_cumsum,
            "global_metrics": global_metrics,
        }



def _reconstruct_train_series_phys(y_train_windows: np.ndarray, scaler_target) -> np.ndarray:
    """
    Reconstruct a 1D train series from windowed data. If it 'looks' scaled (Z-score-ish),
    apply inverse_transform; otherwise assume it's already in physical units.
    """
    from common.seq_preprocessing import reconstruct_true_series
    series = reconstruct_true_series(y_train_windows)

    if _looks_scaled(series):
        series_phys = _inverse_transform_1d(scaler_target, series)
    else:
        series_phys = series
    return series_phys




def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Unified metric computation (called exactly once per series pair).
    Requires your project's `evaluate(y_true, y_pred) -> (r2, smape, mae)`.
    """
    from evaluation.evaluation import evaluate
    r2, smape, mae = evaluate(y_true, y_pred)
    return {"R²": r2, "SMAPE": smape, "MAE": mae}

def _metrics_row(
    *,
    metrics: Dict[str, float],
    well: str,
    method: str,
    category: str,
    kind: str,
    extra_tags: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a row for a metrics dataframe from precomputed metrics (no recompute).
    """
    row = {
        "Well": well,
        "Method": method,
        "Category": category,
        "Kind": kind,
        "R²": metrics.get("R²"),
        "SMAPE": metrics.get("SMAPE"),
        "MAE": metrics.get("MAE"),
    }
    if extra_tags:
        row.update(extra_tags)
    return row



def compute_metrics_to_df_seq(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    well: str,
    method: str,
    category: str
) -> Dict[str, Any]:
    """
    Legacy-compatible wrapper. Prefer building rows via `_metrics_row` with precomputed metrics.
    If used directly, this will compute metrics once via your project's evaluate_return_dict.
    """
    from evaluation.evaluation import evaluate_return_dict
    metrics = evaluate_return_dict(y_test, y_pred)
    metrics.update({
        "Well": well,
        "Method": method,
        "Category": category,
    })
    return metrics


def _style_champions_table(df: pd.DataFrame):
    """
    Aplica apenas cor de fundo e destaque sem alterar o alinhamento.
    """
    df_to_style = df.copy()
    colors = ['#f7f7f7', '#ffffff']
    df_to_style['group'] = df_to_style['well'].astype('category').cat.codes

    categorical_cols = ['architecture_profile']
    for col in categorical_cols:
        if col in df_to_style.columns:
            df_to_style[col] = df_to_style[col].fillna('N/A')

    def alternating_background(row):
        color = colors[row['group'] % len(colors)]
        return [f'background-color: {color}' for _ in row]

    styled_df = df_to_style.style.apply(alternating_background, axis=1)\
        .format({
            'weighted_score': '{:.4f}',
            'val_smape_cum': '{:.4f}',
            'learning_rate': '{:.6f}',
        })

    try:
        styled_df = styled_df.set_caption("Top N Unique Configurations per Well")
    except AttributeError:
        pass

    if hasattr(styled_df, "hide_index"):
        styled_df = styled_df.hide_index()
    if 'group' in df_to_style.columns and hasattr(styled_df, "hide_columns"):
        styled_df = styled_df.hide_columns(['group'])

    hyperparam_style_cols = [
        'physics_strategy', 'data_sample', 'learning_rate', 'lag_window', 'batch_size',
        'epochs', 'architecture_profile'
    ]
    available_hyper_cols = [c for c in hyperparam_style_cols if c in df_to_style.columns]
    if available_hyper_cols:
        styled_df = styled_df.set_properties(
            subset=available_hyper_cols,
            **{'background-color': '#e6fff2', 'color': '#006d2c', 'font-weight': 'bold'}
        )
    return styled_df


def analyze_best_per_architecture(
    master_df: pd.DataFrame,
    metric_to_optimize: str = "weighted_score",
):
    if master_df.empty:
        print("Master DataFrame is empty. Nothing to analyze.")
        return

    from hpo.sort_utils import best_idx_by_group

    print(f"\n{'='*20} 🏆 Architecture Champions League 🏆 {'='*20}")
    print("Showing the single best trial for each architecture, grouped by well.")

    idx = best_idx_by_group(
        master_df,
        group_cols=["well", "architecture"],
        metric=metric_to_optimize,
        lower_is_better=None,
        default_ascending=True,
    )
    best_performers = master_df.loc[idx].copy()

    if best_performers.empty:
        print("No performers found after filtering.")
        return

    display_cols = [
        "well", "architecture", "weighted_score", "val_smape_cum", "val_smape_agg", "trend_degree",
        "architecture_profile", "physics_strategy", "data_sample", "learning_rate", "lag_window", "batch_size",
    ]
    available_cols = [c for c in display_cols if c in best_performers.columns]

    styled_table = _style_champions_table(
        best_performers[available_cols].sort_values(by=["well", metric_to_optimize], kind="mergesort")
    )
    display(styled_table)

    print(f"\n{'='*20} ✅ Architecture Comparison Complete {'='*20}")
    return best_performers


def analyze_holistically(
    master_df: pd.DataFrame,
    metric_to_optimize: str = "weighted_score",
    n_top_per_well: int = 3,
):
    if master_df.empty:
        print("Master DataFrame is empty. Nothing to analyze.")
        return

    from hpo.sort_utils import best_idx_by_group, topk

    print(f"\n{'='*20} 🌐 Holistic Campaign Analysis {'='*20}")

    print(f"\n--- 🏆 Top {n_top_per_well} Champions per Well (by '{metric_to_optimize}') ---")
    KNOWN_HPO_PARAMS = [
        "architecture_profile", "physics_strategy", "data_sample", "learning_rate",
        "lag_window", "batch_size", "epochs",
    ]
    hyper_cols = [col for col in KNOWN_HPO_PARAMS if col in master_df.columns]
    print(f"   (Identifying unique trials based on hyperparameters: {hyper_cols})")

    def _pick(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.drop_duplicates(subset=hyper_cols) if hyper_cols else grp
        return topk(g, metric_to_optimize, int(n_top_per_well), lower_is_better=None, default_ascending=True)

    top_performers = master_df.groupby("well", group_keys=False, sort=False, dropna=False).apply(_pick)

    if top_performers.empty:
        print("No performers found after filtering.")
        return

    display_cols = [
        "well", "architecture", "weighted_score", "val_smape_agg", "val_smape_cum",
        "architecture_profile", "physics_strategy", "epochs", "data_sample", "learning_rate", "lag_window", "batch_size",
    ]
    available_cols = [c for c in display_cols if c in top_performers.columns]
    styled_table = _style_champions_table(top_performers[available_cols])
    display(styled_table)

    print(f"\n--- 🏛️ Average 'Best Score' by Architecture ---")
    best_per_campaign = master_df.loc[
        best_idx_by_group(master_df, ["campaign"], metric_to_optimize, lower_is_better=None, default_ascending=True)
    ]
    arch_perf = (
        best_per_campaign.groupby("architecture")[metric_to_optimize]
        .agg(["mean", "std", "count"])
        .sort_values(by="mean", kind="mergesort")
    )
    print(arch_perf.to_string(float_format="%.2f"))

    print(f"\n--- 🛢️ Winning Architecture per Well ---")
    best_model_per_well = master_df.loc[
        best_idx_by_group(master_df, ["well"], metric_to_optimize, lower_is_better=None, default_ascending=True)
    ]
    well_summary = (
        best_model_per_well[["well", "architecture", metric_to_optimize]]
        .sort_values(by="well", kind="mergesort")
        .set_index("well")
    )
    print(well_summary.to_string(float_format="%.4f"))

    print(f"\n{'='*20} ✅ Holistic Analysis Complete {'='*20}")
    return top_performers

