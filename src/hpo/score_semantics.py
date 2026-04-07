# src/hpo/score_semantics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


RobustKind = Literal["legacy", "neighbor"]


@dataclass(frozen=True)
class ScoreSemantics:
    """
    Captures the semantic meaning behind a score column.

    Example:
      - robust_score produced by legacy pipeline => kind="legacy", canonical="legacy_robust_score"
      - robust_score produced by neighborhood pipeline => kind="neighbor", canonical="neighbor_robust_score"
    """
    kind: Optional[RobustKind]          # None when not applicable
    requested_col: str                 # e.g., "robust_score"
    canonical_col: str                 # e.g., "legacy_robust_score"
    reason: str


def canonical_robust_col(kind: RobustKind) -> str:
    return "legacy_robust_score" if kind == "legacy" else "neighbor_robust_score"


def resolve_score_semantics(score_col: str, *, default_robust_kind: RobustKind = "legacy") -> ScoreSemantics:
    """
    Resolve semantic meaning for a score column name.

    This does NOT rename anything by itself — it's a guardrail + metadata helper.
    """
    c = (score_col or "").strip() or "weighted_score"

    if c == "robust_score":
        kind = default_robust_kind
        return ScoreSemantics(
            kind=kind,
            requested_col=c,
            canonical_col=canonical_robust_col(kind),
            reason=f"robust_score_assumed_{kind}",
        )

    # Not a robust score (or already explicit)
    return ScoreSemantics(kind=None, requested_col=c, canonical_col=c, reason="non_robust_or_explicit")


def ensure_semantic_score_columns(df: pd.DataFrame, semantics: ScoreSemantics) -> pd.DataFrame:
    """
    Ensures semantic alias columns exist, without breaking backward compatibility.

    Rules:
      - If requested_col == "robust_score" and df has robust_score:
          create canonical_col (legacy_robust_score or neighbor_robust_score) if missing.
      - If requested_col is already explicit, no-op.
    """
    out = df.copy()
    if semantics.kind is None:
        return out

    # robust_score must exist to create the semantic alias
    if semantics.requested_col in out.columns and semantics.canonical_col not in out.columns:
        out[semantics.canonical_col] = out[semantics.requested_col]

    return out
