"""
Formation templates for team building and optimisation.

Defines 4-3-3 variants with explicit role slots we can later feed into the
optimizer/TCI logic.
"""
from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class RoleSlot:
    """A single on-field slot with a coarse position and optional role hint."""
    id: str            # unique within a formation (e.g., "ST", "RW", "CM1")
    position_10: str   # coarse position label (GK, CB, RB, LB, CDM, CM, CAM, ST, RW, LW)
    role_hint: str     # text hint for desired profile (e.g., "Poacher", "Box-to-Box")


@dataclass(frozen=True)
class Formation:
    name: str                  # human-readable name
    code: str                  # stable identifier
    slots: List[RoleSlot]      # ordered list of required slots (XI only for now)


# 4-3-3 base with a single CDM (4-3-3 holding)
FORMATION_433_CDM = Formation(
    name="4-3-3 (CDM)",
    code="433_cdm",
    slots=[
        RoleSlot("GK", "GK", "Shot Stopper"),
        RoleSlot("RB", "RB", "Attacking Wingback"),
        RoleSlot("RCB", "CB", "Ball-Playing CB"),
        RoleSlot("LCB", "CB", "Stopper CB"),
        RoleSlot("LB", "LB", "Attacking Wingback"),
        RoleSlot("CDM", "CDM", "Holding Playmaker"),
        RoleSlot("RCM", "CM", "Box-to-Box Mid"),
        RoleSlot("LCM", "CM", "Deep Playmaker"),
        RoleSlot("RW", "RW", "Pace Winger"),
        RoleSlot("LW", "LW", "Pace Winger"),
        RoleSlot("ST", "ST", "All-Round Striker"),
    ],
)

# 4-3-3 with an attacking midfielder (4-3-3 attacking)
FORMATION_433_CAM = Formation(
    name="4-3-3 (CAM)",
    code="433_cam",
    slots=[
        RoleSlot("GK", "GK", "Shot Stopper"),
        RoleSlot("RB", "RB", "Attacking Wingback"),
        RoleSlot("RCB", "CB", "Ball-Playing CB"),
        RoleSlot("LCB", "CB", "Stopper CB"),
        RoleSlot("LB", "LB", "Attacking Wingback"),
        RoleSlot("RCM", "CM", "Box-to-Box Mid"),
        RoleSlot("LCM", "CM", "Deep Playmaker"),
        RoleSlot("CAM", "CAM", "Creative Playmaker"),
        RoleSlot("RW", "RW", "Pace Winger"),
        RoleSlot("LW", "LW", "Pace Winger"),
        RoleSlot("ST", "ST", "All-Round Striker"),
    ],
)


FORMATIONS: Dict[str, Formation] = {
    FORMATION_433_CDM.code: FORMATION_433_CDM,
    FORMATION_433_CAM.code: FORMATION_433_CAM,
}


def get_formation(code: str) -> Formation:
    if code not in FORMATIONS:
        raise ValueError(f"Unknown formation code: {code}")
    return FORMATIONS[code]


def list_formations() -> List[Formation]:
    return list(FORMATIONS.values())
