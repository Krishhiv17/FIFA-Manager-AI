from pathlib import Path
import sys
from typing import Dict, List, Optional
from pprint import pprint
import numpy as np
import pandas as pd
import joblib
import re
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Feature_Engineering.data_prep import (
    BASE_FEATURE_COLS,
    POSITION_FEATURE_COLS,
    TARGET_VALUE,
    TARGET_OVERALL,
    TARGET_POSITION,
)

DATA_PATH = Path("./data/fifa23_clean.csv")
MODELS_DIR = Path("./models")

# ------------ Load data & models once ------------ #

_df: Optional[pd.DataFrame] = None
_value_art = None
_overall_art = None
_position_art = None


def _assert_not_lfs_pointer(path: Path) -> None:
    """
    Detect Git LFS pointer files and raise a clear error before joblib.load
    tries to unpickle garbage text.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required model file missing: {path}")

    with path.open("rb") as f:
        prefix = f.read(200)

    if b"git-lfs.github.com" in prefix:
        raise RuntimeError(
            f"{path} looks like a Git LFS pointer. "
            "Download the real model files with `git lfs pull`."
        )


def _load_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH, low_memory=False)
    return _df


def _load_value_model():
    global _value_art
    if _value_art is None:
        _assert_not_lfs_pointer(MODELS_DIR / "value_xgb.joblib")
        _value_art = joblib.load(MODELS_DIR / "value_xgb.joblib")
    return _value_art


def _load_overall_model():
    global _overall_art
    if _overall_art is None:
        _assert_not_lfs_pointer(MODELS_DIR / "overall_xgb.joblib")
        _overall_art = joblib.load(MODELS_DIR / "overall_xgb.joblib")
    return _overall_art


def _load_position_model():
    global _position_art
    if _position_art is None:
        _assert_not_lfs_pointer(MODELS_DIR / "position_xgb.joblib")
        _position_art = joblib.load(MODELS_DIR / "position_xgb.joblib")
    return _position_art


# ------------ Player lookup helpers ------------ #

def _normalize_name(text: str) -> str:
    """Lowercase, remove weird chars, collapse spaces."""
    text = text.lower()
    text = text.replace(".", " ")
    # keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_players_by_name(query: str, top_k: int = 5) -> pd.DataFrame:
    """
    Fuzzy search on short_name + long_name.
    Handles typos like 'Leonel Messi' vs 'Lionel Messi' and 'L. Messi'.
    Returns top_k best matches.
    """
    df = _load_df()
    q_norm = _normalize_name(query)
    q_tokens = [t for t in q_norm.split() if t]

    # 1) Build a search key per player: "short long"
    name_keys = (
        df["short_name"].fillna("")
        + " "
        + df["long_name"].fillna("")
    ).astype(str)

    name_keys_norm = name_keys.apply(_normalize_name)

    # 2) First try simple substring match on normalized names
    sub_mask = name_keys_norm.str.contains(q_norm, na=False)
    candidates = df[sub_mask].copy()

    # If we got some matches, return best few
    if len(candidates) > 0:
        # to make it deterministic-ish, sort by overall or value
        candidates = candidates.sort_values(
            by=["overall", "value_eur"], ascending=False
        )
        return candidates.head(top_k)

    # 3) Token-overlap match (handles missing middle names, order differences)
    if q_tokens:
        q_token_set = set(q_tokens)
        token_scores = name_keys_norm.apply(
            lambda s: len(q_token_set & set(s.split())) / len(q_token_set)
            if s else 0.0
        )
        token_mask = token_scores > 0
        if token_mask.any():
            cand = df[token_mask].copy()
            cand["__score"] = token_scores[token_mask]
            cand = cand.sort_values(
                by=["__score", "overall", "value_eur"], ascending=False
            )
            return cand.drop(columns="__score").head(top_k)

    # 4) Fuzzy match fallback (for typos etc.)
    # Compute similarity score between query and each name
    scores = []
    for idx, name in enumerate(name_keys_norm):
        if not name:
            continue
        s = SequenceMatcher(None, q_norm, name).ratio()
        scores.append((idx, s))

    # Sort by similarity desc and pick top_k where similarity is decent
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return df.head(0)  # empty

    # You can add a threshold if you want, e.g. > 0.6
    best_indices = [idx for idx, s in scores[:top_k]]
    return df.iloc[best_indices].copy()


def get_player_row_exact(short_name: str) -> Optional[pd.Series]:
    """
    Exact (case-insensitive) match on short_name.
    Returns first match or None if not found.
    """
    df = _load_df()
    mask = df["short_name"].str.lower() == short_name.lower()
    if not mask.any():
        return None
    return df[mask].iloc[0]


# ------------ Prediction helpers ------------ #

def predict_value_for_row(row: pd.Series) -> Dict:
    art = _load_value_model()
    model = art["model"]
    feature_cols = art["feature_cols"]

    X = row[feature_cols].values.astype(float).reshape(1, -1)
    log_pred = model.predict(X)[0]
    value_pred = float(np.expm1(log_pred))

    return {
        "log_value_pred": float(log_pred),
        "value_pred": value_pred,
    }


def predict_overall_for_row(row: pd.Series) -> Dict:
    art = _load_overall_model()
    model = art["model"]
    feature_cols = art["feature_cols"]

    X = row[feature_cols].values.astype(float).reshape(1, -1)
    overall_pred = float(model.predict(X)[0])

    return {
        "overall_pred": overall_pred,
    }


def predict_position_for_row(row: pd.Series) -> Dict:
    art = _load_position_model()
    model = art["model"]
    feature_cols = art["feature_cols"]
    le = art["label_encoder"]

    X = row[feature_cols].values.astype(float).reshape(1, -1)
    class_id = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    label = le.inverse_transform([class_id])[0]
    # top-3 positions
    top_idx = np.argsort(probs)[::-1][:3]
    top_labels = le.inverse_transform(top_idx)
    top_probs = probs[top_idx]

    top3 = [
        {"position": str(lbl), "prob": float(p)}
        for lbl, p in zip(top_labels, top_probs)
    ]

    return {
        "position_pred": str(label),
        "position_top3": top3,
    }


def predict_all_for_player(short_name: str) -> Dict:
    row = get_player_row_exact(short_name)
    if row is None:
        raise ValueError(f"Player '{short_name}' not found in dataset.")

    value_actual = float(row["value_eur"])
    overall_actual = float(row["overall"])
    position_actual = str(row["position_10"])

    value_pred = predict_value_for_row(row)
    overall_pred = predict_overall_for_row(row)
    position_pred = predict_position_for_row(row)

    return {
        "player": {
            "short_name": str(row["short_name"]),
            "long_name": str(row["long_name"]),
            "age": int(row["age"]),
            "potential": float(row.get("potential", overall_actual)),  # new
            "club_name": str(row.get("club_name", "")),
            "nationality_name": str(row.get("nationality_name", "")),
            "pace": float(row["pace"]),
            "shooting": float(row["shooting"]),
            "dribbling": float(row["dribbling"]),
            "defending": float(row["defending"]),
            "physic": float(row["physic"]),
        },
        "actual": {
            "value_eur": value_actual,
            "overall": overall_actual,
            "position_10": position_actual,
        },
        "predictions": {
            "value": value_pred,
            "overall": overall_pred,
            "position": position_pred,
        },
    }



if __name__ == "__main__":
    # Simple manual test
    df = _load_df()
    example_name = df["short_name"].iloc[0]
    print(f"Testing predictions for: {example_name}")
    result = predict_all_for_player(example_name)
    pprint(result)
