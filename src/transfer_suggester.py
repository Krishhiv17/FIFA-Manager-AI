"""
Transfer suggestion pipeline (skeleton).

Given a pre-built team (list of players with positions) and user constraints,
suggest potential transfers/upgrades using the player pool.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.team_optimizer import (
    TeamConstraints,
    load_player_pool,
    score_player_for_slot,
    style_presets,
)
from src.formations import get_formation


@dataclass
class CurrentPlayer:
    short_name: str
    position_10: str  # assigned position
    playstyle: Optional[str] = None
    overall: Optional[float] = None
    value_eur: Optional[float] = None


def normalize_current_team(raw_team: List[Dict]) -> List[CurrentPlayer]:
    """
    Convert user-provided team dicts into CurrentPlayer objects.
    Expected keys per item: short_name, position_10 (assigned), optional playstyle/overall/value_eur.
    """
    normalized = []
    for item in raw_team:
        normalized.append(
            CurrentPlayer(
                short_name=str(item.get("short_name")),
                position_10=str(item.get("position_10", "")).upper(),
                playstyle=item.get("playstyle"),
                overall=float(item["overall"]) if "overall" in item else None,
                value_eur=float(item["value_eur"]) if "value_eur" in item else None,
            )
        )
    return normalized


def _acceptable_positions(slot_pos: str) -> List[str]:
    """Mirror the slot-compatibility used in the optimizer (simple fallbacks)."""
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


def suggest_transfers(
    current_team: List[Dict],
    constraints: TeamConstraints,
    max_suggestions: int = 3,
    target_positions: Optional[List[str]] = None,
    replace_players: Optional[List[str]] = None,
    max_age: Optional[int] = None,
    min_pace: Optional[float] = None,
    min_physic: Optional[float] = None,
) -> List[Dict]:
    """
    For each slot in the chosen formation, if the current player is below
    a simple threshold vs pool alternatives, suggest upgrades.
    """
    norm_team = normalize_current_team(current_team)
    formation = get_formation(constraints.formation_code)
    slots = [s for s in formation.slots if s.position_10 != "GK"]  # skip GK suggestions
    pool = load_player_pool()

    suggestions: List[Dict] = []
    preset = style_presets(constraints.style or "balanced")
    # Apply style-based playstyle prefs if not provided
    if not constraints.prefer_playstyles:
        constraints.prefer_playstyles = preset.get("prefer_playstyles", [])
    eff_min_overall = constraints.min_overall if constraints.min_overall is not None else preset.get("min_overall")
    eff_min_pace = min_pace if min_pace is not None else preset.get("min_pace")
    eff_min_physic = min_physic if min_physic is not None else preset.get("min_physic")
    eff_min_passing = constraints.min_passing if constraints.min_passing is not None else preset.get("min_passing")
    # normalize current names for duplicate checks
    current_names = {p.short_name.strip().lower() for p in norm_team}

    used_currents = set()

    for slot in slots:
        if target_positions and slot.position_10 not in target_positions:
            continue
        # Find current player for this slot by exact position match, avoid reusing same current twice
        allowed_positions = _acceptable_positions(slot.position_10)
        current = next((p for p in norm_team if p.position_10 == slot.position_10 and p.short_name not in used_currents), None)
        if replace_players and current and current.short_name not in replace_players:
            continue
        if current:
            used_currents.add(current.short_name)

        # Build a score threshold: if current exists, we try to beat their score by a margin
        current_score = None
        if current:
            match = pool[pool["short_name"] == current.short_name]
            if not match.empty:
                current_score = score_player_for_slot(match.iloc[0], slot, constraints.prefer_playstyles)
            else:
                current_score = 50.0  # unknown player baseline

        # Score pool candidates for this slot (position match only)
        scored = []
        for _, row in pool.iterrows():
            name_norm = str(row.get("short_name", "")).strip().lower()
            if name_norm in current_names:
                continue  # don't suggest players already in the squad
            if str(row.get("position_10", "")).upper() not in allowed_positions:
                continue
            if max_age is not None and row.get("age", 0) > max_age:
                continue
            if eff_min_pace is not None and row.get("pace", 0) < eff_min_pace:
                continue
            if eff_min_physic is not None and row.get("physic", 0) < eff_min_physic:
                continue
            if eff_min_passing is not None and row.get("passing", 0) < eff_min_passing:
                continue
            if eff_min_overall is not None and row.get("overall", 0) < eff_min_overall:
                continue
            cand_score = score_player_for_slot(row, slot, constraints.prefer_playstyles)
            scored.append((cand_score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Pick top candidates better than current by a margin (or top if no current)
        for cand_score, row in scored:
            if current_score is not None and cand_score < current_score + 5:
                continue  # not a meaningful upgrade

            suggestion = {
                "slot": slot.id,
                "desired_role": slot.role_hint,
                "current_player": current.short_name if current else None,
                "candidate": {
                    "short_name": row["short_name"],
                    "long_name": row["long_name"],
                    "position_10": row["position_10"],
                    "playstyle": row.get("playstyle"),
                    "overall": row["overall"],
                    "value_eur": row["value_eur"],
                    "pace": row["pace"],
                    "physic": row["physic"],
                },
                "score_gain": cand_score - (current_score or 0.0),
            }
            suggestions.append(suggestion)
            break  # only top upgrade per slot for now

        if len(suggestions) >= max_suggestions:
            break

    # Sort suggestions by score gain descending
    suggestions.sort(key=lambda s: s["score_gain"], reverse=True)
    return suggestions[:max_suggestions]


def suggest_replacements_for_player(
    current_player: Dict[str, str],
    constraints: TeamConstraints,
    desired_playstyles: Optional[List[str]] = None,
    max_suggestions: int = 3,
) -> List[Dict]:
    """
    Suggest up to max_suggestions replacements for a single player, based on position,
    style constraints, and desired playstyles. Uses per-player budget cap if provided.
    """
    pool = load_player_pool()
    preset = style_presets(constraints.style or "balanced")
    prefer_styles = desired_playstyles or []
    if not prefer_styles:
        prefer_styles = preset.get("prefer_playstyles", [])

    # Infer position if not provided
    target_name = str(current_player.get("short_name", "")).strip()
    target_pos = str(current_player.get("position_10", "")).strip().upper()
    if not target_pos:
        row = pool[pool["short_name"].str.lower() == target_name.lower()]
        if not row.empty:
            target_pos = str(row.iloc[0].get("position_10", "")).upper()
    if not target_pos:
        return []

    allowed_positions = _acceptable_positions(target_pos)

    eff_min_overall = constraints.min_overall if constraints.min_overall is not None else preset.get("min_overall")
    eff_min_pace = constraints.min_pace if constraints.min_pace is not None else preset.get("min_pace")
    eff_min_physic = constraints.min_physic if constraints.min_physic is not None else preset.get("min_physic")
    eff_min_passing = constraints.min_passing if constraints.min_passing is not None else preset.get("min_passing")

    # Budget per player (use provided budget_eur as a hard cap for this replacement)
    per_player_cap = float(constraints.budget_eur) if constraints.budget_eur is not None else None

    def role_hint_for_pos(pos: str) -> str:
        pos = pos.upper()
        if pos in {"RB", "LB"}:
            return "Attacking Wingback"
        if pos == "CB":
            return "Stopper CB"
        if pos == "CDM":
            return "Holding Playmaker"
        if pos == "CM":
            return "Box-to-Box Mid"
        if pos == "CAM":
            return "Creative Playmaker"
        if pos in {"RW", "LW"}:
            return "Pace Winger"
        if pos == "ST":
            return "Poacher"
        return pos

    dummy_slot = type(
        "Slot",
        (),
        {"position_10": target_pos, "role_hint": role_hint_for_pos(target_pos)},
    )()

    scored = []
    current_norm = target_name.lower()
    for _, row in pool.iterrows():
        name_norm = str(row.get("short_name", "")).strip().lower()
        if name_norm == current_norm:
            continue
        pos = str(row.get("position_10", "")).upper()
        if pos not in allowed_positions:
            continue
        if constraints.max_age is not None and row.get("age", 0) > constraints.max_age:
            continue
        if eff_min_overall is not None and row.get("overall", 0) < eff_min_overall:
            continue
        if eff_min_pace is not None and row.get("pace", 0) < eff_min_pace:
            continue
        if eff_min_physic is not None and row.get("physic", 0) < eff_min_physic:
            continue
        if eff_min_passing is not None and row.get("passing", 0) < eff_min_passing:
            continue
        if per_player_cap is not None and row.get("value_eur", 0) > per_player_cap:
            continue
        score = score_player_for_slot(row, dummy_slot, prefer_styles)
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    suggestions = []
    for score, row in scored[:max_suggestions]:
        suggestions.append(
            {
                "current_player": target_name,
                "slot": target_pos,
                "candidate": {
                    "short_name": row["short_name"],
                    "long_name": row["long_name"],
                    "position_10": row["position_10"],
                    "playstyle": row.get("playstyle"),
                    "overall": row["overall"],
                    "value_eur": row["value_eur"],
                    "pace": row["pace"],
                    "physic": row["physic"],
                },
                "score": score,
            }
        )
    return suggestions


if __name__ == "__main__":
    # Interactive prompt: enter current players as "short_name:position_10" lines, blank to finish.
    print("=== Transfer Suggester ===")
    style = input("Team style (balanced/counter/high_press/possession): ").strip().lower() or "balanced"
    formation_code = input("Formation (433_cdm/433_cam): ").strip() or "433_cdm"

    current_team = []
    print("Enter your current players as 'short_name:POSITION_10' (e.g., 'L. Messi:RW'). Blank line to finish.")
    while True:
        line = input("> ").strip()
        if not line:
            break
        if ":" not in line:
            print("Format should be short_name:POSITION_10 (e.g., L. Messi:RW)")
            continue
        name, pos = line.split(":", 1)
        current_team.append({"short_name": name.strip(), "position_10": pos.strip().upper()})

    print("Optional: target positions to improve (comma-separated, e.g., ST,RW) or blank for all:")
    target_positions = [p.strip().upper() for p in input("> ").split(",") if p.strip()] or None
    print("Optional: specific players to replace (short_name, comma-separated) or blank:")
    replace_players = [p.strip() for p in input("> ").split(",") if p.strip()] or None
    print("Optional: max age (blank for none):")
    max_age = input("> ").strip()
    max_age = int(max_age) if max_age else None

    constraints = TeamConstraints(style=style, formation_code=formation_code)
    suggestions = suggest_transfers(
        current_team=current_team,
        constraints=constraints,
        max_suggestions=5,
        target_positions=target_positions,
        replace_players=replace_players,
        max_age=max_age,
    )
    if not suggestions:
        print("No suggestions found under current constraints.")
    else:
        for s in suggestions:
            c = s["candidate"]
            print(f"{s['slot']}: Suggest {c['short_name']} ({c.get('playstyle','')}) "
                  f"overall {c['overall']} value €{c['value_eur']:,.0f} "
                  f"(replacing {s['current_player']})")
