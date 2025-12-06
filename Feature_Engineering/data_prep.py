from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import re

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Path to your raw dataset. Adjust if needed.
RAW_DATA_PATH = Path("./data/player_stats.csv")

# Position rating columns in FIFA (contain strings like "87+3", "63+2", "0+0")
POS_RATING_COLS: List[str] = [
    "ls", "st", "rs",
    "lw", "lf", "cf", "rf", "rw",
    "lam", "cam", "ram",
    "lm", "lcm", "cm", "rcm", "rm",
    "lwb", "ldm", "cdm", "rdm", "rwb",
    "lb", "lcb", "cb", "rcb", "rb",
    "gk",
]

# Our 10-position label set
POSITION_10_LABELS = ["CB", "RB", "LB", "GK", "CM", "CDM", "CAM", "ST", "RW", "LW"]

# Core numeric feature set (shared across all three tasks)
BASE_FEATURE_COLS: List[str] = [
    # Bio
    "age",
    "height_cm",
    "weight_kg",
    # Aggregate FIFA stats
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    # Attacking / skill
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    # Movement
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    # Power
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    # Mentality
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    # Defending details
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    # GK stats (critical for GK; mostly noise for outfield, but the model will learn that)
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]

POSITION_FEATURE_COLS = BASE_FEATURE_COLS + [
    "overall",        # EA's overall – very indicative of role tier
    "log_value_eur",  # value signal – attackers/GKs sometimes differ
]

TARGET_VALUE = "value_eur"
TARGET_OVERALL = "overall"
TARGET_POSITION = "position_10"


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def load_raw_df(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle FIFA players file."""
    df = pd.read_csv(path, low_memory=False)
    return df


def filter_fifa23(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only FIFA 23 rows (latest fifa_version in this file)."""
    # In this dataset, fifa_version runs from 15 to 23.
    # We'll keep only version 23.
    df23 = df[df["fifa_version"] == 23].copy()
    df23.reset_index(drop=True, inplace=True)
    return df23


def parse_pos_rating(val) -> float:
    """
    Parse strings like '87+3', '92-1', '0+0' into numeric base ratings:
    - '87+3' -> 87
    - '92-1' -> 92
    - '0+0'  -> 0
    If parsing fails or val is NaN, returns np.nan.
    """
    if pd.isna(val):
        return np.nan
    s = str(val)
    # Extract leading digits
    m = re.match(r"(\d+)", s)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except ValueError:
        return np.nan


def add_numeric_position_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the positional rating columns from strings like '87+3'
    to numeric base ratings in-place.
    """
    for col in POS_RATING_COLS:
        if col in df.columns:
            df[col] = df[col].apply(parse_pos_rating)
        else:
            # If a col is missing for some reason, create it as NaN
            df[col] = np.nan
    return df


def extract_primary_position_from_player_positions(player_positions: str) -> str:
    """
    FIFA 'player_positions' is like "RW,ST,LW" or "CB".
    Take the FIRST listed position as the primary.
    """
    if pd.isna(player_positions):
        return None
    parts = [p.strip() for p in str(player_positions).split(",") if p.strip()]
    return parts[0] if parts else None


def map_to_position_10(detailed_pos: str) -> str:
    """
    Map a detailed FIFA position (ST, CF, LWB, RDM, etc.)
    to one of our 10 coarse labels: CB, RB, LB, GK, CM, CDM, CAM, ST, RW, LW.
    """
    if detailed_pos is None:
        return None

    p = detailed_pos.upper()

    # Goalkeeper
    if p == "GK":
        return "GK"

    # Centre-backs
    if p in ["CB", "LCB", "RCB"]:
        return "CB"

    # Fullbacks / wingbacks
    if p in ["RB", "RWB"]:
        return "RB"
    if p in ["LB", "LWB"]:
        return "LB"

    # Defensive / central mids
    if p in ["CDM", "LDM", "RDM"]:
        return "CDM"
    if p in ["CM", "LCM", "RCM"]:
        return "CM"

    # Attacking mids / central forwards
    if p in ["CAM", "LAM", "RAM"]:
        return "CAM"
    if p in ["CF", "LF", "RF"]:
        return "ST"  # treat second strikers as STs

    # Wingers
    if p in ["RW", "RM"]:
        return "RW"
    if p in ["LW", "LM"]:
        return "LW"

    # Strikers
    if p in ["ST", "LS", "RS"]:
        return "ST"

    # If we don't recognise it, just return the raw value (will be rare)
    return p


def derive_best_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a 'best_position_detailed' from the positional rating columns,
    then map it to our coarse 'position_10' label.

    Logic:
      - use argmax over numeric positional ratings (ls/st/.../gk)
      - if all ratings are NaN or zero, fall back to player_positions primary
    """
    # Ensure numeric ratings are present
    df = add_numeric_position_ratings(df)

    # Argmax over position rating columns
    pos_matrix = df[POS_RATING_COLS].values
    # Handle rows where everything is NaN by temporarily filling with -1
    pos_matrix_nan_to_minus1 = np.where(np.isnan(pos_matrix), -1, pos_matrix)
    best_idx = pos_matrix_nan_to_minus1.argmax(axis=1)
    best_scores = pos_matrix_nan_to_minus1.max(axis=1)

    # Build detailed best position from rating-based argmax
    best_pos_from_ratings = [
        POS_RATING_COLS[i] if score >= 0 else None
        for i, score in zip(best_idx, best_scores)
    ]

    df["best_position_detailed"] = best_pos_from_ratings

    # Fallback: if best_position_detailed is None or rating is 0,
    # use the first entry in player_positions
    primary_from_str = df["player_positions"].apply(
        extract_primary_position_from_player_positions
    )

    use_fallback_mask = df["best_position_detailed"].isna() | (
        df[POS_RATING_COLS].max(axis=1) <= 0
    )
    df.loc[use_fallback_mask, "best_position_detailed"] = primary_from_str[use_fallback_mask]

    # Map to coarse 10-class label
    df["position_10"] = df["best_position_detailed"].apply(map_to_position_10)

    return df


# ---------------------------------------------------------------------
# MAIN PREPROCESS PIPELINE
# ---------------------------------------------------------------------

def preprocess_df(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Full cleaning & feature engineering pipeline:

      - load CSV
      - filter to FIFA 23
      - parse positional ratings into numeric
      - derive 'best_position_detailed' and 'position_10'
      - drop rows with missing/invalid targets & features
      - ensure numeric types for feature cols
    """
    df = load_raw_df(path)
    df = filter_fifa23(df)
    df = derive_best_position(df)

    # Drop rows with missing essential columns
    cols_needed = BASE_FEATURE_COLS + [TARGET_VALUE, TARGET_OVERALL, TARGET_POSITION]
    df = df.dropna(subset=cols_needed).copy()

    # Convert features to float
    df[BASE_FEATURE_COLS] = df[BASE_FEATURE_COLS].astype(float)
    df[TARGET_VALUE] = df[TARGET_VALUE].astype(float)
    df[TARGET_OVERALL] = df[TARGET_OVERALL].astype(float)

    # Filter out obviously weird rows
    df = df[df[TARGET_VALUE] > 0].reset_index(drop=True)

    # Add log-transformed value for regression stability
    df["log_value_eur"] = np.log1p(df[TARGET_VALUE])

    return df


# ---------------------------------------------------------------------
# DATASET GETTERS
# ---------------------------------------------------------------------

def get_value_dataset(path: Path = RAW_DATA_PATH) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return X, y (log(value_eur)), and the cleaned df for market value prediction."""
    df = preprocess_df(path)
    X = df[BASE_FEATURE_COLS].values
    y = df["log_value_eur"].values
    return X, y, df


def get_overall_dataset(path: Path = RAW_DATA_PATH) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return X, y (overall), and the cleaned df for overall rating prediction."""
    df = preprocess_df(path)
    X = df[BASE_FEATURE_COLS].values
    y = df[TARGET_OVERALL].values
    return X, y, df


def get_position_dataset(path: Path = RAW_DATA_PATH) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Return X, y (position_10), and df for **position classification**.

    NOTE: We deliberately DO NOT include the positional rating columns as features here,
    otherwise we'd be leaking the label. We only use core attributes + overall + log_value.
    """
    df = preprocess_df(path)

    X = df[POSITION_FEATURE_COLS].values
    y = df[TARGET_POSITION].values
    return X, y, df



if __name__ == "__main__":
    df_clean = preprocess_df()
    print("Cleaned dataset shape:", df_clean.shape)
    print("\nPosition_10 distribution:")
    print(df_clean["position_10"].value_counts())

    # Save cleaned version once
    out_path = Path("./data/fifa23_clean.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"\n[Saved] Cleaned dataset → {out_path.resolve()}")

