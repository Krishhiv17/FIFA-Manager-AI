"""
Rule-based playstyle tagging for FIFA players.

This module derives an interpretable `playstyle` label for each player using
their attributes, positions, traits, and tags. It favors deterministic,
transparent rules rather than ML so we can tweak thresholds easily later.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Feature_Engineering.data_prep import (
    preprocess_df,
    extract_primary_position_from_player_positions,
    map_to_position_10,
)


# Stats we rely on for thresholds; we compute percentiles so rules adapt to the dataset.
STATS_FOR_THRESHOLDS: List[str] = [
    # Core
    "pace",
    "dribbling",
    "passing",
    "shooting",
    "defending",
    "physic",
    # Attacking details
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_crossing",
    "skill_dribbling",
    "skill_long_passing",
    "mentality_positioning",
    "mentality_vision",
    # Movement / power
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "power_strength",
    "power_jumping",
    "power_stamina",
    "power_shot_power",
    "power_long_shots",
    # Defending / mentality
    "mentality_interceptions",
    "defending_standing_tackle",
    "defending_marking_awareness",
    "mentality_aggression",
    # GK
    "goalkeeping_reflexes",
    "goalkeeping_diving",
    "goalkeeping_speed",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
]


def _primary_pos10(row: pd.Series) -> Optional[str]:
    """Derive the coarse 10-position label from the first listed position."""
    primary = extract_primary_position_from_player_positions(
        row.get("player_positions")
    )
    return map_to_position_10(primary) if primary else None


def _has_trait(row: pd.Series, needle: str) -> bool:
    traits = str(row.get("player_traits") or "")
    return needle.lower() in traits.lower()


def _has_tag(row: pd.Series, needle: str) -> bool:
    tags = str(row.get("player_tags") or "")
    return needle.lower() in tags.lower()


def _build_percentiles(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, Dict[float, float]]:
    """Precompute percentiles for the columns we use in rules."""
    quantiles = [0.5, 0.7, 0.85]
    out: Dict[str, Dict[float, float]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        out[col] = {q: float(series.quantile(q)) for q in quantiles}
    return out


def _is_high(row: pd.Series, col: str, pct: float, ptiles: Dict[str, Dict[float, float]]) -> bool:
    if col not in ptiles:
        return False
    try:
        return float(row.get(col, np.nan)) >= ptiles[col].get(pct, np.inf)
    except Exception:
        return False


def _is_low(row: pd.Series, col: str, pct: float, ptiles: Dict[str, Dict[float, float]]) -> bool:
    if col not in ptiles:
        return False
    try:
        val = float(row.get(col, np.nan))
    except Exception:
        return False
    return val <= ptiles[col].get(pct, -np.inf)


def _choose_playstyle(row: pd.Series, ptiles: Dict[str, Dict[float, float]]) -> str:
    pos10 = _primary_pos10(row)
    traits = str(row.get("player_traits") or "")
    tags = str(row.get("player_tags") or "")

    # Goalkeepers
    if pos10 == "GK":
        if _is_high(row, "goalkeeping_reflexes", 0.85, ptiles) and _is_high(
            row, "goalkeeping_diving", 0.85, ptiles
        ):
            return "GK Shot Stopper"
        if _is_high(row, "goalkeeping_speed", 0.85, ptiles) or _has_trait(
            row, "1-on-1 Rush"
        ):
            return "GK Sweeper"
        if _is_high(row, "goalkeeping_kicking", 0.85, ptiles):
            return "GK Distributor"
        return "GK Balanced"

    # Wingers / wide forwards
    if pos10 in {"RW", "LW"}:
        if _has_tag(row, "speedster") or (
            _is_high(row, "pace", 0.85, ptiles) and _is_high(row, "dribbling", 0.7, ptiles)
        ):
            return "Pace Winger"
        if _is_high(row, "attacking_crossing", 0.7, ptiles) and _is_high(
            row, "power_stamina", 0.7, ptiles
        ):
            return "Wide Workhorse"
        if _is_high(row, "passing", 0.7, ptiles) and _is_high(
            row, "mentality_vision", 0.7, ptiles
        ):
            return "Wide Playmaker"
        return "Balanced Winger"

    # Strikers / central forwards
    if pos10 == "ST":
        if _is_high(row, "attacking_finishing", 0.85, ptiles) and _is_high(
            row, "mentality_positioning", 0.7, ptiles
        ):
            if _is_high(row, "pace", 0.7, ptiles):
                return "Poacher"
            return "Penalty Box Forward"
        if _is_high(row, "power_strength", 0.85, ptiles) and _is_high(
            row, "attacking_heading_accuracy", 0.7, ptiles
        ):
            return "Target Forward"
        if _is_high(row, "passing", 0.7, ptiles) and _is_high(
            row, "mentality_vision", 0.7, ptiles
        ):
            return "False Nine"
        return "All-Round Striker"

    # Attacking mids
    if pos10 == "CAM":
        if _is_high(row, "dribbling", 0.85, ptiles) and _is_high(
            row, "passing", 0.7, ptiles
        ):
            return "Creative Playmaker"
        if _is_high(row, "shooting", 0.7, ptiles) and _is_high(
            row, "power_shot_power", 0.7, ptiles
        ):
            return "Attacking Mid Shooter"
        return "Balanced CAM"

    # Central mids
    if pos10 == "CM":
        if _is_high(row, "power_stamina", 0.85, ptiles) and _is_high(
            row, "pace", 0.7, ptiles
        ):
            return "Box-to-Box Mid"
        if _is_high(row, "passing", 0.85, ptiles) and _is_high(
            row, "mentality_vision", 0.7, ptiles
        ):
            return "Deep Playmaker"
        if _is_high(row, "defending", 0.7, ptiles) and _is_high(
            row, "mentality_interceptions", 0.7, ptiles
        ):
            return "Ball-Winning Mid"
        return "Balanced CM"

    # Defensive mids
    if pos10 == "CDM":
        if _is_high(row, "mentality_interceptions", 0.85, ptiles) and _is_high(
            row, "mentality_aggression", 0.7, ptiles
        ):
            return "Destroyer CDM"
        if _is_high(row, "passing", 0.7, ptiles) and _is_high(
            row, "defending", 0.7, ptiles
        ):
            return "Holding Playmaker"
        if _is_high(row, "power_strength", 0.7, ptiles) and _is_high(
            row, "defending_standing_tackle", 0.7, ptiles
        ):
            return "Anchor CDM"
        return "Balanced CDM"

    # Fullbacks / wingbacks
    if pos10 in {"RB", "LB"}:
        if _is_high(row, "pace", 0.85, ptiles) and _is_high(
            row, "attacking_crossing", 0.7, ptiles
        ):
            return "Attacking Wingback"
        if _is_high(row, "defending", 0.7, ptiles) and _is_high(
            row, "power_strength", 0.7, ptiles
        ):
            return "Defensive Fullback"
        if _is_high(row, "power_stamina", 0.85, ptiles):
            return "Two-Way Fullback"
        return "Balanced Fullback"

    # Centre-backs
    if pos10 == "CB":
        if _is_high(row, "power_strength", 0.85, ptiles) and _is_high(
            row, "attacking_heading_accuracy", 0.7, ptiles
        ):
            return "Stopper CB"
        if _is_high(row, "passing", 0.7, ptiles) and _is_high(
            row, "defending_marking_awareness", 0.7, ptiles
        ):
            return "Ball-Playing CB"
        return "Balanced CB"

    # Wide mids mapped to RW/LW should already be handled; fallback for unknowns
    return "Unspecified"


def add_playstyle_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with a new column `playstyle` computed via rules.
    """
    ptiles = _build_percentiles(df, STATS_FOR_THRESHOLDS)
    df_out = df.copy()
    df_out["playstyle"] = df_out.apply(lambda row: _choose_playstyle(row, ptiles), axis=1)
    return df_out


def save_with_playstyle(
    input_df: Optional[pd.DataFrame] = None,
    out_path: Path = Path("./data/fifa23_with_playstyle.csv"),
) -> Path:
    """
    Convenience helper: load cleaned dataset (or use provided df), add playstyle,
    and save to CSV.
    """
    if input_df is None:
        input_df = preprocess_df()
    df_with_style = add_playstyle_column(input_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_style.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = save_with_playstyle()
    print(f"[Saved] dataset with playstyle → {path.resolve()}")
