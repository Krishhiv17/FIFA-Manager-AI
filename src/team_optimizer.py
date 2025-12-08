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
import random

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formations import Formation, RoleSlot, get_formation
from Feature_Engineering.data_prep import (
    extract_primary_position_from_player_positions,
    map_to_position_10,
)

# Default data path (enriched with playstyle)
PLAYER_DATA_PATH = Path("./data/fifa23_clean_with_playstyle.csv")


@dataclass
class TeamConstraints:
    formation_code: str = "433_cdm"
    style: str = "balanced"
    budget_eur: Optional[float] = None  # total squad budget cap
    gk_reserve_pct: float = 0.1         # reserve share of budget for GK (we skip GK selection)
    max_age: Optional[int] = None
    min_pace: Optional[float] = None
    min_physic: Optional[float] = None
    min_stamina: Optional[float] = None
    min_passing: Optional[float] = None
    min_overall: Optional[float] = None
    prefer_playstyles: List[str] = field(default_factory=list)
    mandatory_players: List[str] = field(default_factory=list)  # short_name(s) to include
    blocked_players: List[str] = field(default_factory=list)    # short_name(s) to exclude
    bench_size: int = 3


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
        "pace", "physic", "age", "playstyle", "player_positions",
        "passing", "shooting", "defending", "power_stamina",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in dataset {path}")

    # Normalize position_10; prefer cleaned labels but fall back to map
    df["position_10"] = df["position_10"].astype(str).str.strip().str.upper()
    valid_positions = {"GK", "CB", "RB", "LB", "CDM", "CM", "CAM", "ST", "RW", "LW"}

    def _rederive_pos(row):
        pos = str(row.get("position_10", "")).strip().upper()
        if pos in valid_positions:
            return pos
        primary = extract_primary_position_from_player_positions(row.get("player_positions"))
        mapped = map_to_position_10(primary) if primary else None
        return mapped if mapped else "CB"

    df["position_10"] = df.apply(_rederive_pos, axis=1)

    return df


def _find_player_by_short_name(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    mask = df["short_name"].astype(str).str.lower() == name.lower()
    if not mask.any():
        return None
    return df[mask].iloc[0]


# Playstyle preferences per role hint (used in scoring)
ROLE_PLAYSTYLE_PREFS: Dict[str, List[str]] = {
    "Attacking Wingback": ["Attacking Wingback", "Two-Way Fullback"],
    "Ball-Playing CB": ["Ball-Playing CB"],
    "Stopper CB": ["Stopper CB", "Balanced CB"],
    "Holding Playmaker": ["Holding Playmaker", "Anchor CDM", "Destroyer CDM"],
    "Box-to-Box Mid": ["Box-to-Box Mid"],
    "Deep Playmaker": ["Deep Playmaker", "Creative Playmaker"],
    "Creative Playmaker": ["Creative Playmaker", "Wide Playmaker"],
    "Pace Winger": ["Pace Winger", "Wide Playmaker", "Balanced Winger"],
    "All-Round Striker": ["All-Round Striker", "Poacher", "Target Forward"],
    "Target Forward": ["Target Forward", "All-Round Striker"],
    "Poacher": ["Poacher", "All-Round Striker"],
    "GK Shot Stopper": ["GK Shot Stopper"],
    "GK Sweeper": ["GK Sweeper", "GK Distributor"],
}


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


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
    alias = {
        "stamina": "power_stamina",
    }
    col_real = alias.get(col, col)
    try:
        return float(row.get(col_real, 0.0))
    except Exception:
        return 0.0


def _pos10(row: pd.Series) -> str:
    return str(row.get("position_10", "")).strip().upper()


def _acceptable_positions(slot_pos: str) -> List[str]:
    """Return allowed position_10 values for a slot, with simple mirroring fallbacks."""
    slot_pos = slot_pos.upper()
    if slot_pos == "RB":
        return ["RB", "LB"]
    if slot_pos == "LB":
        return ["LB", "RB"]
    if slot_pos == "RW":
        return ["RW", "LW", "ST"]
    if slot_pos == "LW":
        return ["LW", "RW", "ST"]
    if slot_pos == "CAM":
        return ["CAM", "CM", "CDM"]
    if slot_pos == "CDM":
        return ["CDM", "CM"]
    if slot_pos == "CM":
        return ["CM", "CDM", "CAM"]
    if slot_pos == "CB":
        return ["CB", "LB", "RB"]
    if slot_pos == "ST":
        return ["ST", "LW", "RW"]
    return [slot_pos]


def _compute_effective_budget(constraints: TeamConstraints) -> Optional[float]:
    if constraints.budget_eur is None:
        return None
    try:
        budget_val = float(constraints.budget_eur)
    except Exception:
        return None
    reserve = max(0.0, min(0.5, constraints.gk_reserve_pct))
    return budget_val * (1 - reserve)


def prefilter_pool(df: pd.DataFrame, constraints: TeamConstraints) -> pd.DataFrame:
    """
    Global prefilter to shrink search space based on style/constraints.
    Applies age and style-based floors before slot-specific selection.
    """
    preset = style_presets(constraints.style or "balanced")
    eff_min_overall = constraints.min_overall if constraints.min_overall is not None else preset.get("min_overall")
    eff_min_pace = constraints.min_pace if constraints.min_pace is not None else preset.get("min_pace")
    eff_min_physic = constraints.min_physic if constraints.min_physic is not None else preset.get("min_physic")
    eff_min_passing = constraints.min_passing if constraints.min_passing is not None else preset.get("min_passing")

    pos10 = df["position_10"].astype(str).str.upper()
    is_gk = pos10 == "GK"

    mask = pd.Series([True] * len(df))
    if constraints.max_age is not None:
        mask &= df["age"] <= constraints.max_age
    # Apply style floors only to non-GK
    nongk_mask = ~is_gk
    if eff_min_overall is not None:
        mask &= (~nongk_mask) | (df["overall"] >= eff_min_overall)
    if eff_min_pace is not None:
        mask &= (~nongk_mask) | (df["pace"] >= eff_min_pace)
    if eff_min_physic is not None:
        mask &= (~nongk_mask) | (df["physic"] >= eff_min_physic)
    if eff_min_passing is not None:
        mask &= (~nongk_mask) | (df["passing"] >= eff_min_passing)

    filtered = df[mask].copy()

    # Keep best row per short_name by overall to reduce duplicates
    if not filtered.empty:
        filtered.sort_values(by=["short_name", "overall"], ascending=[True, False], inplace=True)
        filtered = filtered.drop_duplicates(subset=["short_name"], keep="first")

    return filtered


def _make_player_entry(row: pd.Series, slot_id: str, role_hint: str) -> Dict:
    return {
        "slot": slot_id,
        "role_hint": role_hint,
        "position_10": str(row.get("position_10", "")).upper(),
        "player": {
            "short_name": row.get("short_name"),
            "long_name": row.get("long_name"),
            "age": row.get("age"),
            "overall": row.get("overall"),
            "value_eur": row.get("value_eur"),
            "playstyle": row.get("playstyle"),
            "position_10": row.get("position_10"),
            "pace": row.get("pace"),
            "physic": row.get("physic"),
            "passing": row.get("passing"),
            "shooting": row.get("shooting"),
            "defending": row.get("defending"),
            "stamina": row.get("power_stamina"),
        },
        "score": _stat(row, "overall"),
    }


def select_goalkeeper(
    df: pd.DataFrame,
    constraints: TeamConstraints,
    used_names: set,
    current_value: float,
    total_budget: Optional[float],
) -> Tuple[Optional[Dict], float]:
    gks = df[df["position_10"].str.upper() == "GK"].copy()
    if constraints.max_age is not None:
        gks = gks[gks["age"] <= constraints.max_age]
    if constraints.min_overall is not None:
        gks = gks[gks["overall"] >= constraints.min_overall]
    gks = gks[~gks["short_name"].str.lower().isin(used_names)]

    if gks.empty:
        return None, 0.0

    gks = gks.sort_values(by=["overall", "value_eur"], ascending=[False, True])

    if total_budget is not None:
        remaining = total_budget - current_value
        affordable = gks[gks["value_eur"] <= remaining]
        if not affordable.empty:
            row = affordable.iloc[0]
            return _make_player_entry(row, slot_id="GK", role_hint="GK Shot Stopper"), _stat(row, "value_eur")
        # If nothing fits, fall through to pick the cheapest GK to at least fill the slot
        gks = gks.sort_values(by=["value_eur", "overall"], ascending=[True, False])
        row = gks.iloc[0]
        return _make_player_entry(row, slot_id="GK", role_hint="GK Shot Stopper"), _stat(row, "value_eur")

    row = gks.iloc[0]
    return _make_player_entry(row, slot_id="GK", role_hint="GK Shot Stopper"), _stat(row, "value_eur")


def _playstyle_fit_bonus(playstyle: str, role_hint: str, prefer_playstyles: List[str]) -> float:
    ps_lower = playstyle.lower()
    bonus = 0.0

    # Role-specific preference
    role_pref = ROLE_PLAYSTYLE_PREFS.get(role_hint, [])
    if any(ps_lower == rp.lower() for rp in role_pref):
        bonus += 6.0
    elif role_pref and any(rp.lower() in ps_lower for rp in role_pref):
        bonus += 3.0

    # User preferred styles
    if prefer_playstyles and any(p.lower() in ps_lower for p in prefer_playstyles):
        bonus += 4.0

    return bonus


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

    ps = str(row.get("playstyle", ""))
    score += _playstyle_fit_bonus(ps, slot.role_hint, prefer_playstyles)

    # General overall boost
    score += 0.5 * _stat(row, "overall")
    return score


def _passes_filters(
    row: pd.Series,
    constraints: TeamConstraints,
    current_value: float,
    slots_left: int,
    slot_position: Optional[str] = None,
    effective_budget: Optional[float] = None,
) -> bool:
    pos = (slot_position or _pos10(row) or "").strip().upper()
    if constraints.max_age is not None and row.get("age", 0) > constraints.max_age:
        return False
    # For GKs, do not enforce outfield athletic minima
    if pos != "GK":
        if constraints.min_pace is not None and _stat(row, "pace") < constraints.min_pace:
            return False
        if constraints.min_physic is not None and _stat(row, "physic") < constraints.min_physic:
            return False
        if constraints.min_stamina is not None and _stat(row, "stamina") < constraints.min_stamina:
            return False
        if constraints.min_passing is not None and _stat(row, "passing") < constraints.min_passing:
            return False
    if constraints.min_overall is not None and _stat(row, "overall") < constraints.min_overall:
        return False
    if constraints.blocked_players and str(row.get("short_name")) in constraints.blocked_players:
        return False

    # Budget enforcement with GK reserve
    if effective_budget is not None:
        # simple per-slot affordability: ensure we can afford this player plus at least 0 for remaining slots
        if current_value + _stat(row, "value_eur") > effective_budget:
            return False
        # also apply a soft ceiling per remaining slot (budget / slots_left) to avoid one player blowing budget
        if slots_left > 0:
            max_per_slot = effective_budget / (slots_left + 1e-9)
            if _stat(row, "value_eur") > max_per_slot * 2:  # allow some flexibility
                return False
    return True


def _outfield_slots(formation: Formation) -> List[RoleSlot]:
    """Return formation slots excluding GK."""
    return [s for s in formation.slots if s.position_10 != "GK"]


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
    used_names = set()
    # compute effective budget once
    effective_budget = None
    if constraints.budget_eur is not None:
        try:
            budget_val = float(constraints.budget_eur)
            reserve = max(0.0, min(0.5, constraints.gk_reserve_pct))
            effective_budget = budget_val * (1 - reserve)
        except Exception:
            effective_budget = None
    team: List[Dict] = []
    total_value = 0.0

    slots = _outfield_slots(formation)

    # Pre-assign mandatory players to matching outfield slots if possible
    for m in constraints.mandatory_players:
        row = _find_player_by_short_name(df, m)
        if row is None:
            continue
        target_pos = str(row.get("position_10", "")).upper()
        available_slots = [s for s in slots if s.id not in {t["slot"] for t in team}]
        slot = next((s for s in available_slots if s.position_10 == target_pos), None)
        if slot is None:
            slot = next(
                (s for s in available_slots if target_pos in _acceptable_positions(s.position_10)),
                None,
            )
        if slot is None:
            continue
        # ignore budget for mandatory, but respect age/blocked/min overall
        if constraints.blocked_players and str(row.get("short_name")) in constraints.blocked_players:
            continue
        if constraints.max_age is not None and row.get("age", 0) > constraints.max_age:
            continue
        if constraints.min_overall is not None and _stat(row, "overall") < constraints.min_overall:
            continue
        idx = row.name
        used.add(idx)
        used_names.add(str(row.get("short_name", "")).strip().lower())
        total_value += _stat(row, "value_eur")
        team.append(
            {
                "slot": slot.id,
                "role_hint": slot.role_hint,
                "position_10": slot.position_10,
                "player": {
                    "short_name": row.get("short_name"),
                    "long_name": row.get("long_name"),
                    "age": row.get("age"),
                    "overall": row.get("overall"),
                    "value_eur": row.get("value_eur"),
                    "playstyle": row.get("playstyle"),
                    "position_10": row.get("position_10"),
                    "pace": row.get("pace"),
                    "physic": row.get("physic"),
                    "passing": row.get("passing"),
                    "shooting": row.get("shooting"),
                    "defending": row.get("defending"),
                    "stamina": row.get("power_stamina"),
                },
                "score": 999.0,  # force-keep mandatory player
            }
        )

    for slot in slots:
        if any(t["slot"] == slot.id for t in team):
            continue  # already filled by mandatory

        slots_left = len(slots) - len(team) - 1
        allowed_positions = _acceptable_positions(slot.position_10)

        # Progressive relaxation if strict filters yield nothing (keep overall floor)
        overall_floor = constraints.min_overall
        relax_levels = [
            {"min_pace": constraints.min_pace, "min_physic": constraints.min_physic, "min_overall": overall_floor},
            {"min_pace": None, "min_physic": constraints.min_physic, "min_overall": overall_floor},
            {"min_pace": None, "min_physic": None, "min_overall": overall_floor},
        ]

        candidates = []
        for relax in relax_levels:
            temp_constraints = TeamConstraints(
                formation_code=constraints.formation_code,
                style=constraints.style,
                budget_eur=constraints.budget_eur,
                gk_reserve_pct=constraints.gk_reserve_pct,
                max_age=constraints.max_age,
                min_pace=relax["min_pace"],
                min_physic=relax["min_physic"],
                min_stamina=constraints.min_stamina,
                min_passing=constraints.min_passing,
                min_overall=relax["min_overall"],
                prefer_playstyles=constraints.prefer_playstyles,
                mandatory_players=constraints.mandatory_players,
                blocked_players=constraints.blocked_players,
                bench_size=constraints.bench_size,
            )
            for idx, row in df.iterrows():
                if idx in used:
                    continue
                short_name = str(row.get("short_name"))
                short_name_norm = short_name.strip().lower()
                if temp_constraints.blocked_players and short_name in temp_constraints.blocked_players:
                    continue
                if short_name_norm in used_names:
                    continue
                if _pos10(row) not in allowed_positions:
                    continue
                if not _passes_filters(
                    row,
                    temp_constraints,
                    total_value,
                    slots_left,
                    slot_position=slot.position_10,
                    effective_budget=effective_budget,
                ):
                    continue
                score = score_player_for_slot(row, slot, temp_constraints.prefer_playstyles)
                candidates.append((score, idx, row))
            if candidates:
                break

        if not candidates:
            # last-resort: pick a candidate ignoring budget but respecting position/overall/blocked
            budget_relaxed = []
            for idx, row in df.iterrows():
                if idx in used:
                    continue
                name_norm = str(row.get("short_name", "")).strip().lower()
                if constraints.blocked_players and str(row.get("short_name")) in constraints.blocked_players:
                    continue
                if name_norm in used_names:
                    continue
                if _pos10(row) not in allowed_positions:
                    continue
                if overall_floor is not None and _stat(row, "overall") < overall_floor:
                    continue
                # Use inverse value to prefer cheaper when budget-relaxed
                score = _stat(row, "overall") - 0.0000001 * _stat(row, "value_eur")
                budget_relaxed.append((score, idx, row))
            candidates = budget_relaxed
        if not candidates:
            raise ValueError(f"No available candidates for slot {slot.id} under current constraints.")

        # pick best-scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_row = candidates[0]
        # If adding this player blows the budget, skip and try next candidate
        if effective_budget is not None and total_value + _stat(best_row, "value_eur") > effective_budget:
            # try next best
            placed = False
            for cand_score, cand_idx, cand_row in candidates[1:]:
                if effective_budget is not None and total_value + _stat(cand_row, "value_eur") > effective_budget:
                    continue
                best_score, best_idx, best_row = cand_score, cand_idx, cand_row
                placed = True
                break
            if not placed:
                raise ValueError(f"No affordable candidates for slot {slot.id} under budget constraints.")

        used.add(best_idx)
        used_names.add(str(best_row.get("short_name", "")).strip().lower())
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
                    "position_10": best_row.get("position_10"),
                    "pace": best_row.get("pace"),
                    "physic": best_row.get("physic"),
                    "passing": best_row.get("passing"),
                    "shooting": best_row.get("shooting"),
                    "defending": best_row.get("defending"),
                    "stamina": best_row.get("power_stamina"),
                },
                "score": best_score,
            }
        )

    avg_overall = sum(_stat(item["player"], "overall") for item in team) / len(team)
    starter_positions = {item["position_10"] for item in team}
    effective_budget = None
    if constraints.budget_eur is not None:
        reserve = max(0.0, min(0.5, constraints.gk_reserve_pct))
        effective_budget = constraints.budget_eur * (1 - reserve)
    starter_value = total_value
    bench, bench_value = select_bench(
        df,
        used,
        constraints,
        starter_positions=starter_positions,
        starter_avg_overall=avg_overall,
        current_total_value=starter_value,
    )
    # Enforce budget on bench: trim most expensive bench players until within budget
    if effective_budget is not None and starter_value + bench_value > effective_budget:
        bench_sorted = sorted(bench, key=lambda b: _stat(b["player"], "value_eur"), reverse=True)
        while bench_sorted and starter_value + bench_value > effective_budget:
            removed = bench_sorted.pop(0)
            bench_value -= _stat(removed["player"], "value_eur")
        bench = bench_sorted
    total_value += bench_value

    summary = {
        "formation": formation.name,
        "total_value_eur": total_value,
        "avg_overall": avg_overall,
        "slots": len(team),
        "bench": bench,
    }
    return team, summary


def _build_candidate_pool(df: pd.DataFrame, formation: Formation, constraints: TeamConstraints) -> Dict[str, List[int]]:
    pools: Dict[str, List[int]] = {}
    for slot in _outfield_slots(formation):
        idxs = []
        allowed_positions = _acceptable_positions(slot.position_10)
        for idx, row in df.iterrows():
            if constraints.blocked_players and str(row.get("short_name")) in constraints.blocked_players:
                continue
            # enforce position match for pools with fallback
            if _pos10(row) not in allowed_positions:
                continue
            # Relax budget in pool; we'll handle budget in fitness.
            if not _passes_filters(row, constraints, current_value=0.0, slots_left=0, slot_position=slot.position_10):
                continue
            idxs.append(idx)
        pools[slot.id] = idxs
    return pools


def _assemble_team_from_indices(df: pd.DataFrame, slots: List[RoleSlot], idx_by_slot: Dict[str, int]) -> List[Dict]:
    team = []
    for slot in slots:
        idx = idx_by_slot[slot.id]
        row = df.loc[idx]
        team.append(
            {
                "slot": slot.id,
                "role_hint": slot.role_hint,
                "position_10": slot.position_10,
                "player": {
                    "short_name": row.get("short_name"),
                    "long_name": row.get("long_name"),
                    "age": row.get("age"),
                    "overall": row.get("overall"),
                    "value_eur": row.get("value_eur"),
                    "playstyle": row.get("playstyle"),
                    "position_10": row.get("position_10"),
                    "pace": row.get("pace"),
                    "physic": row.get("physic"),
                    "passing": row.get("passing"),
                    "shooting": row.get("shooting"),
                    "defending": row.get("defending"),
                    "stamina": row.get("power_stamina"),
                },
                "score": 0.0,
            }
        )
    return team


def select_bench(
    df: pd.DataFrame,
    used_indices: set,
    constraints: TeamConstraints,
    starter_positions: Optional[set] = None,
    starter_avg_overall: float = 0.0,
    current_total_value: float = 0.0,
) -> Tuple[List[Dict], float]:
    """
    Simple bench selection: pick highest scoring versatile players not in XI,
    aiming to cover missing positions.
    """
    bench = []
    total_value = 0.0
    bench_positions = set()
    starter_positions = starter_positions or set()

    candidates = []
    used_names = {str(df.loc[idx].get("short_name", "")).strip().lower() for idx in used_indices}
    for idx, row in df.iterrows():
        if idx in used_indices:
            continue
        name_norm = str(row.get("short_name", "")).strip().lower()
        if name_norm in used_names:
            continue
        if _pos10(row) == "GK":
            continue  # ignore GK for bench selection
        if not _passes_filters(row, constraints, current_value=0.0, slots_left=0, slot_position=_pos10(row)):
            continue
        pos_str = [p.strip().upper() for p in str(row.get("player_positions", "")).split(",") if p.strip()]
        versatility = len(pos_str)
        base_score = _stat(row, "overall") + 0.1 * versatility
        # Penalize very expensive bench players
        base_score -= 0.0000005 * _stat(row, "value_eur")
        # Penalize if much higher than starters (keep stars on XI)
        if starter_avg_overall:
            base_score -= max(0.0, (_stat(row, "overall") - starter_avg_overall) * 0.5)
        candidates.append((base_score, idx, row))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for _, idx, row in candidates:
        if len(bench) >= constraints.bench_size:
            break
        pos10 = _pos10(row)
        # Prefer covering starter positions not yet covered in bench
        needs_coverage = starter_positions - bench_positions
        bonus = 7.0 if pos10 in needs_coverage else (3.0 if pos10 not in bench_positions else 0.0)
        bench.append(
            {
                "slot": f"BENCH{len(bench)+1}",
                "position_10": pos10,
                "player": {
                    "short_name": row.get("short_name"),
                    "long_name": row.get("long_name"),
                    "age": row.get("age"),
                    "overall": row.get("overall"),
                    "value_eur": row.get("value_eur"),
                    "playstyle": row.get("playstyle"),
                    "position_10": pos10,
                    "pace": row.get("pace"),
                    "physic": row.get("physic"),
                    "passing": row.get("passing"),
                    "shooting": row.get("shooting"),
                    "defending": row.get("defending"),
                    "stamina": row.get("power_stamina"),
                },
                "score": _stat(row, "overall") + bonus,
            }
        )
        total_value += _stat(row, "value_eur")
        bench_positions.add(pos10)
        used_names.add(name_norm)

    return bench, total_value


def _fitness_team(team: List[Dict], constraints: TeamConstraints, df: Optional[pd.DataFrame] = None, used_indices: Optional[set] = None) -> Tuple[float, Dict]:
    """
    Fitness based on TCI minus penalties for budget, duplicates, missing mandatory.
    """
    bench = []
    bench_value = 0.0
    starter_positions = {item["position_10"] for item in team}
    if df is not None and used_indices is not None and constraints.bench_size > 0:
        bench, bench_value = select_bench(df, used_indices, constraints, starter_positions=starter_positions)

    tci = compute_tci(team, constraints, bench=bench)
    fitness = tci["tci"]

    # Budget handling
    total_value = sum(_stat(item["player"], "value_eur") for item in team) + bench_value
    effective_budget = _compute_effective_budget(constraints)

    # Hard cap: discard over-budget individuals
    if effective_budget is not None and total_value > effective_budget:
        return -1e9, {"tci": tci, "total_value": total_value, "bench": bench, "bench_value": bench_value}

    # Small bonus for budget utilization (encourages spending up to cap)
    if effective_budget is not None:
        utilization = total_value / (effective_budget + 1e-9)
        fitness += 5.0 * utilization  # capped by utilization

    # Penalty for duplicate players
    names = [item["player"]["short_name"] for item in team]
    dup_penalty = (len(names) - len(set(names))) * 30.0
    fitness -= dup_penalty

    # Penalty for missing mandatory players
    if constraints.mandatory_players:
        missing = [m for m in constraints.mandatory_players if m not in names]
        fitness -= len(missing) * 25.0

    return fitness, {"tci": tci, "total_value": total_value, "bench": bench, "bench_value": bench_value}


def ga_select_team(
    df: pd.DataFrame,
    formation: Formation,
    constraints: TeamConstraints,
    generations: int = 12,
    pop_size: int = 24,
    mutation_prob: float = 0.2,
) -> Tuple[List[Dict], Dict]:
    """
    Simple genetic algorithm to search team combinations.
    """
    effective_budget = _compute_effective_budget(constraints)
    pools = _build_candidate_pool(df, formation, constraints)
    if any(len(v) == 0 for v in pools.values()):
        raise ValueError("No candidates for at least one slot under given constraints.")

    def random_team_indices() -> Dict[str, int]:
        return {slot.id: random.choice(pools[slot.id]) for slot in _outfield_slots(formation)}

    def crossover(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        child = {}
        for i, slot in enumerate(_outfield_slots(formation)):
            child[slot.id] = a[slot.id] if i % 2 == 0 else b[slot.id]
        return child

    def mutate(ind: Dict[str, int]) -> Dict[str, int]:
        mutant = dict(ind)
        for slot in _outfield_slots(formation):
            if random.random() < mutation_prob:
                mutant[slot.id] = random.choice(pools[slot.id])
        return mutant

    slots = _outfield_slots(formation)
    population = [random_team_indices() for _ in range(pop_size)]
    best_team = None
    best_fit = -1e9
    best_meta = {}

    for _ in range(generations):
        scored_pop = []
        for ind in population:
            team = _assemble_team_from_indices(df, slots, ind)
            used = set(ind.values())
            fit, meta = _fitness_team(team, constraints, df=df, used_indices=used)
            scored_pop.append((fit, ind, meta, team))
            if fit > best_fit:
                best_fit = fit
                best_team = team
                best_meta = meta

        # Selection: top-k elitism + tournament
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        elites = [ind for _, ind, _, _ in scored_pop[:4]]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            a = random.choice(scored_pop[:10])[1]
            b = random.choice(scored_pop[:10])[1]
            child = crossover(a, b)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    if best_team is None:
        raise ValueError("Genetic search could not find a feasible team within the given budget/constraints.")

    # Add GK recommendation
    used_names = {str(item["player"].get("short_name", "")).strip().lower() for item in best_team}
    gk_entry, gk_value = select_goalkeeper(
        df,
        constraints,
        used_names,
        best_meta.get("total_value", 0.0),
        float(constraints.budget_eur) if constraints.budget_eur is not None else None,
    )
    team_with_gk = list(best_team)
    total_value_with_gk = best_meta.get("total_value", 0.0)
    if gk_entry:
        team_with_gk.append(gk_entry)
        total_value_with_gk += gk_value

    bench = best_meta.get("bench", [])
    avg_overall = sum(_stat(item["player"], "overall") for item in team_with_gk) / len(team_with_gk)

    summary = {
        "formation": formation.name,
        "total_value_eur": total_value_with_gk,
        "avg_overall": avg_overall,
        "slots": len(team_with_gk),
        "tci": best_meta.get("tci", {}),
        "bench": bench,
    }
    return team_with_gk, summary


def compute_tci(team: List[Dict], constraints: TeamConstraints, bench: Optional[List[Dict]] = None) -> Dict[str, float]:
    """
    Team Competitiveness Index: self-contained composite on 0-100 scale.
    Components:
      - positional_fit: are players on their natural coarse positions?
      - role_playstyle_fit: alignment of playstyle with slot role hints + user prefs.
      - style_compliance: how well the team meets style preset mins (pace/physic/stamina/passing/overall).
      - line_balance: quick synergy check across lines (defense solidity, attack threat).
      - depth: bench coverage and average quality (if bench provided).
    """
    if not team:
        return {"tci": 0.0}

    # Positional fit
    pos_scores = []
    for item in team:
        player_pos = str(item["player"].get("position_10", "")).upper()
        slot_pos = item["position_10"]
        pos_scores.append(100.0 if player_pos == slot_pos else 40.0)
    positional_fit = sum(pos_scores) / len(pos_scores)

    # Role / playstyle fit
    role_scores = []
    for item in team:
        role_hint = item["role_hint"]
        ps = str(item["player"].get("playstyle", ""))
        role_pref = ROLE_PLAYSTYLE_PREFS.get(role_hint, [])
        if any(ps.lower() == rp.lower() for rp in role_pref):
            role_scores.append(100.0)
        elif any(rp.lower() in ps.lower() for rp in role_pref):
            role_scores.append(70.0)
        elif constraints.prefer_playstyles and any(p.lower() in ps.lower() for p in constraints.prefer_playstyles):
            role_scores.append(60.0)
        else:
            role_scores.append(40.0)
    role_playstyle_fit = sum(role_scores) / len(role_scores)

    # Style compliance using preset minima
    preset = style_presets(constraints.style or "balanced")
    style_checks = []
    for key, stat_col in [
        ("min_pace", "pace"),
        ("min_physic", "physic"),
        ("min_stamina", "stamina"),
        ("min_passing", "passing"),
        ("min_overall", "overall"),
    ]:
        min_req = preset.get(key)
        if min_req is None:
            continue
        ok = sum(1 for item in team if _stat(item["player"], stat_col) >= min_req)
        style_checks.append(100.0 * ok / len(team))
    style_compliance = sum(style_checks) / len(style_checks) if style_checks else 80.0

    # Line balance / synergy (rough heuristics)
    defense = [item for item in team if item["position_10"] in {"CB", "RB", "LB", "CDM"}]
    attack = [item for item in team if item["position_10"] in {"ST", "RW", "LW"} or item["role_hint"].lower().startswith("creative")]
    mid = [item for item in team if item["position_10"] in {"CM", "CAM"}]

    def _avg_stat(items, col):
        return sum(_stat(it["player"], col) for it in items) / len(items) if items else 0.0

    defense_solid = (_avg_stat(defense, "defending") + _avg_stat(defense, "physic")) / 2
    attack_threat = (_avg_stat(attack, "pace") + _avg_stat(attack, "shooting")) / 2
    build_play = (_avg_stat(mid, "passing") + _avg_stat(mid, "pace")) / 2

    # Normalize to 0-100 assuming raw stats roughly 0-100
    line_balance = (defense_solid + attack_threat + build_play) / 3

    # Depth component (bench)
    depth_score = 75.0
    if bench:
        starter_positions = {item["position_10"] for item in team if item["position_10"] != "GK"}
        bench_positions = {item["position_10"] for item in bench if item["position_10"] != "GK"}
        coverage_ratio = len(bench_positions & starter_positions) / (len(starter_positions) + 1e-9)
        bench_overall = sum(_stat(item["player"], "overall") for item in bench) / len(bench)
        starter_overall = sum(_stat(item["player"], "overall") for item in team) / len(team)
        bench_vs_starter = bench_overall / (starter_overall + 1e-9) * 100
        depth_score = 0.7 * coverage_ratio * 100 + 0.3 * bench_vs_starter

    # Composite weighting
    tci = (
        0.3 * positional_fit
        + 0.25 * role_playstyle_fit
        + 0.25 * style_compliance
        + 0.15 * line_balance
        + 0.05 * depth_score
    )

    return {
        "tci": _clamp(tci),
        "positional_fit": _clamp(positional_fit),
        "role_playstyle_fit": _clamp(role_playstyle_fit),
        "style_compliance": _clamp(style_compliance),
        "line_balance": _clamp(line_balance),
        "depth": _clamp(depth_score),
    }


def style_presets(style: str) -> Dict:
    """
    Map a team style keyword to constraint defaults.
    Styles inspired by common FIFA tactics: counter, high_press, possession, balanced.
    """
    s = style.lower()
    if s in {"counter", "counter_attack"}:
        return {
            "min_pace": 80,
            "min_physic": 68,
            "min_overall": 75,
            "prefer_playstyles": ["Pace Winger", "Attacking Wingback", "All-Round Striker"],
        }
    if s in {"high_press", "press"}:
        return {
            "min_pace": 75,
            "min_physic": 70,
            "min_stamina": 75,
            "min_overall": 75,
            "prefer_playstyles": ["Box-to-Box Mid", "Attacking Wingback", "Pace Winger", "All-Round Striker"],
        }
    if s in {"possession", "tiki_taka"}:
        return {
            "min_pace": 70,
            "min_physic": 65,
            "min_passing": 75,
            "min_overall": 78,
            "prefer_playstyles": ["Creative Playmaker", "Deep Playmaker", "Ball-Playing CB"],
        }
    # balanced default
    return {}


def _input_float(prompt: str) -> Optional[float]:
    val = input(prompt).strip()
    if not val:
        return None
    try:
        clean = val.replace(",", "").replace("_", "")
        return float(clean)
    except ValueError:
        print("Invalid number, ignoring.")
        return None


def _input_list(prompt: str) -> List[str]:
    val = input(prompt).strip()
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def build_constraints_interactive() -> TeamConstraints:
    print("Choose formation code (433_cdm, 433_cam). Press Enter for default 433_cdm.")
    formation = input("> Formation: ").strip() or "433_cdm"
    print("Choose team style: balanced, counter, high_press, possession. Press Enter for balanced.")
    style = input("> Style: ").strip().lower() or "balanced"
    preset = style_presets(style)

    print("Budget cap in euros (blank for none):")
    budget = _input_float("> Budget: ")
    max_age = _input_float("> Max age (blank for none): ")
    min_overall = _input_float("> Min overall (blank for preset/none): ")
    min_pace = _input_float("> Min pace (blank for preset/none): ")
    min_physic = _input_float("> Min physic (blank for preset/none): ")

    prefer_styles = preset.get("prefer_playstyles", [])
    extra_pref = _input_list("> Preferred playstyles (comma-separated, blank for none): ")
    if extra_pref:
        prefer_styles += extra_pref

    mandatory = _input_list("> Mandatory players (short_name, comma-separated, blank for none): ")
    blocked = _input_list("> Blocked players (short_name, comma-separated, blank for none): ")

    gk_reserve_pct = _input_float("> Reserve % of budget for GK (0-0.5, blank defaults to 0.1): ")
    if gk_reserve_pct is None:
        gk_reserve_pct = 0.1
    gk_reserve_pct = max(0.0, min(0.5, gk_reserve_pct))

    return TeamConstraints(
        formation_code=formation,
        style=style,
        budget_eur=budget,
        gk_reserve_pct=gk_reserve_pct,
        max_age=int(max_age) if max_age is not None else None,
        min_pace=min_pace if min_pace is not None else preset.get("min_pace"),
        min_physic=min_physic if min_physic is not None else preset.get("min_physic"),
        min_stamina=preset.get("min_stamina"),
        min_passing=preset.get("min_passing"),
        min_overall=min_overall if min_overall is not None else preset.get("min_overall"),
        prefer_playstyles=prefer_styles,
        mandatory_players=mandatory,
        blocked_players=blocked,
    )


if __name__ == "__main__":
    print("=== Team Optimizer (genetic only) ===")
    pool = load_player_pool()
    constraints = build_constraints_interactive()
    formation = get_formation(constraints.formation_code)
    pool = prefilter_pool(pool, constraints)

    team, summary = ga_select_team(pool, formation, constraints)
    tci = summary.get("tci", compute_tci(team, constraints, bench=summary.get("bench")))

    print(f"\nFormation: {summary['formation']} | Style: {constraints.style} | Mode: genetic")
    print(f"Avg overall: {summary['avg_overall']:.2f} | Total value: €{summary['total_value_eur']:,.0f}")
    print(f"TCI: {tci['tci']:.1f} "
          f"(positional {tci['positional_fit']:.1f}, role/playstyle {tci['role_playstyle_fit']:.1f}, "
          f"style compliance {tci['style_compliance']:.1f}, line balance {tci['line_balance']:.1f}, depth {tci['depth']:.1f})")
    for item in team:
        p = item["player"]
        print(
            f"{item['slot']:4s} -> {p['short_name']} ({p.get('playstyle','')}) "
            f"| overall {p['overall']} | value €{p['value_eur']:,.0f}"
        )
    if summary.get("bench"):
        print("\nBench:")
        for b in summary["bench"]:
            p = b["player"]
            print(
                f"{b['slot']:6s} -> {p['short_name']} ({p.get('playstyle','')}) "
                f"| pos {p.get('position_10','')} | overall {p['overall']} | value €{p['value_eur']:,.0f}"
            )
