"""
Baseline team optimizer scaffolding.

Implements:
- Constraint schema (budget, age/pace/physic/overall minima, playstyle prefs).
- Greedy selector for 4-3-3 variants defined in `src/formations.py`.

This is an intentionally simple first pass; later we can swap the search with
GA/CP-SAT once scoring pieces are richer.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formations import Formation, get_formation

# Default data path (enriched with playstyle)
PLAYER_DATA_PATH = Path("./data/fifa23_with_playstyle.csv")


@dataclass
class TeamConstraints:
    formation_code: str = "433_cdm"
    budget_eur: Optional[float] = None  # total squad budget cap
    max_age: Optional[int] = None
    min_pace: Optional[float] = None
    min_physic: Optional[float] = None
    min_overall: Optional[float] = None
    prefer_playstyles: List[str] = field(default_factory=list)
    mandatory_players: List[str] = field(default_factory=list)  # short_name(s) to include
    blocked_players: List[str] = field(default_factory=list)    # short_name(s) to exclude


def load_player_pool(path: Path = PLAYER_DATA_PATH) -> pd.DataFrame:
    """
    Load the player dataset with playstyle. Falls back to raising if missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Player dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    # Basic sanity: ensure needed cols exist
    required_cols = [
        "short_name", "long_name", "position_10", "overall", "value_eur",
        "pace", "physic", "age", "playstyle",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset {path}")
    return df


def _role_hint_weights(role_hint: str) -> Dict[str, float]:
    """
    Lightweight weight presets per role hint to guide scoring.
    """
    rh = role_hint.lower()
    if "wingback" in rh:
        return {"pace": 0.25, "physic": 0.1, "defending": 0.2, "attacking_crossing": 0.15, "stamina": 0.1}
    if "cb" in rh:
        return {"defending": 0.35, "physic": 0.2, "power_strength": 0.15, "attacking_heading_accuracy": 0.1}
    if "playmaker" in rh:
        return {"passing": 0.35, "mentality_vision": 0.2, "dribbling": 0.15}
    if "box-to-box" in rh:
        return {"stamina": 0.25, "pace": 0.15, "defending": 0.15, "passing": 0.15}
    if "destroyer" in rh or "holding" in rh or "cdm" in rh:
        return {"defending": 0.3, "mentality_interceptions": 0.2, "physic": 0.15, "stamina": 0.1}
    if "winger" in rh:
        return {"pace": 0.35, "dribbling": 0.25, "attacking_crossing": 0.1}
    if "striker" in rh or "forward" in rh:
        return {"attacking_finishing": 0.35, "pace": 0.2, "shooting": 0.15, "mentality_positioning": 0.1}
    if "gk" in rh:
        return {"goalkeeping_reflexes": 0.4, "goalkeeping_diving": 0.2, "goalkeeping_positioning": 0.15}
    return {"overall": 0.3, "pace": 0.15, "passing": 0.15, "physic": 0.15}


def _stat(row: pd.Series, col: str) -> float:
    try:
        return float(row.get(col, 0.0))
    except Exception:
        return 0.0


def score_player_for_slot(row: pd.Series, slot, prefer_playstyles: List[str]) -> float:
    """
    Heuristic scoring: position fit + weighted stats + playstyle preference bonus.
    """
    score = 0.0
    if str(row.get("position_10", "")).upper() == slot.position_10:
        score += 25.0
    else:
        score -= 20.0

    weights = _role_hint_weights(slot.role_hint)
    for col, w in weights.items():
        score += w * _stat(row, col)

    if prefer_playstyles:
        ps = str(row.get("playstyle", "")).lower()
        if ps and any(p.lower() in ps for p in prefer_playstyles):
            score += 5.0

    # General overall boost
    score += 0.5 * _stat(row, "overall")
    return score


def _passes_filters(row: pd.Series, constraints: TeamConstraints, current_value: float, slots_left: int) -> bool:
    if constraints.max_age is not None and row.get("age", 0) > constraints.max_age:
        return False
    if constraints.min_pace is not None and _stat(row, "pace") < constraints.min_pace:
        return False
    if constraints.min_physic is not None and _stat(row, "physic") < constraints.min_physic:
        return False
    if constraints.min_overall is not None and _stat(row, "overall") < constraints.min_overall:
        return False
    if constraints.blocked_players and str(row.get("short_name")) in constraints.blocked_players:
        return False

    if constraints.budget_eur is not None:
        # simple per-slot affordability: ensure we can afford this player plus at least 0 for remaining slots
        if current_value + _stat(row, "value_eur") > constraints.budget_eur:
            return False
        # also apply a soft ceiling per remaining slot (budget / slots_left) to avoid one player blowing budget
        if slots_left > 0:
            max_per_slot = constraints.budget_eur / (slots_left + 1e-9)
            if _stat(row, "value_eur") > max_per_slot * 2:  # allow some flexibility
                return False
    return True


def greedy_select_team(
    df: pd.DataFrame,
    formation: Formation,
    constraints: TeamConstraints,
) -> Tuple[List[Dict], Dict]:
    """
    Greedy XI selection: for each slot, pick the highest scoring affordable player not yet used.
    Returns (team_list, summary_stats).
    """
    used = set()
    team: List[Dict] = []
    total_value = 0.0

    for slot in formation.slots:
        slots_left = len(formation.slots) - len(team) - 1
        candidates = []
        for idx, row in df.iterrows():
            if idx in used:
                continue
            short_name = str(row.get("short_name"))
            if constraints.mandatory_players and short_name in constraints.blocked_players:
                continue
            if constraints.blocked_players and short_name in constraints.blocked_players:
                continue
            if not _passes_filters(row, constraints, total_value, slots_left):
                continue
            score = score_player_for_slot(row, slot, constraints.prefer_playstyles)
            candidates.append((score, idx, row))

        if not candidates:
            raise ValueError(f"No available candidates for slot {slot.id} under current constraints.")

        # pick best-scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_row = candidates[0]
        used.add(best_idx)
        total_value += _stat(best_row, "value_eur")

        team.append(
            {
                "slot": slot.id,
                "role_hint": slot.role_hint,
                "position_10": slot.position_10,
                "player": {
                    "short_name": best_row.get("short_name"),
                    "long_name": best_row.get("long_name"),
                    "age": best_row.get("age"),
                    "overall": best_row.get("overall"),
                    "value_eur": best_row.get("value_eur"),
                    "playstyle": best_row.get("playstyle"),
                },
                "score": best_score,
            }
        )

    avg_overall = sum(_stat(item["player"], "overall") for item in team) / len(team)
    summary = {
        "formation": formation.name,
        "total_value_eur": total_value,
        "avg_overall": avg_overall,
        "slots": len(team),
    }
    return team, summary


if __name__ == "__main__":
    # Simple manual run to test greedy selection
    pool = load_player_pool()
    constraints = TeamConstraints(
        formation_code="433_cdm",
        budget_eur=None,
        max_age=34,
        min_overall=78,
        prefer_playstyles=["Pace Winger"],
    )
    formation = get_formation(constraints.formation_code)
    team, summary = greedy_select_team(pool, formation, constraints)
    print(summary)
    for item in team:
        p = item["player"]
        print(f"{item['slot']:4s} -> {p['short_name']} ({p['playstyle']}) | overall {p['overall']} | value €{p['value_eur']:,.0f}")
