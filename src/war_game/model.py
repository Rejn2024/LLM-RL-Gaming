"""Core domain model for the simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Faction(str, Enum):
    BLUE = "blue"
    RED = "red"
    WHITE = "white"


class Terrain(str, Enum):
    OPEN = "open"
    FOREST = "forest"
    URBAN = "urban"
    WATER = "water"
    HILL = "hill"


TERRAIN_COST = {
    Terrain.OPEN: 1.0,
    Terrain.FOREST: 1.8,
    Terrain.URBAN: 1.5,
    Terrain.WATER: float("inf"),
    Terrain.HILL: 2.0,
}
TERRAIN_DEFENCE = {
    Terrain.OPEN: 0.0,
    Terrain.FOREST: 0.25,
    Terrain.URBAN: 0.35,
    Terrain.WATER: 0.0,
    Terrain.HILL: 0.2,
}


@dataclass(frozen=True, order=True)
class Hex:
    """Axial hex coordinate."""

    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r

    def distance(self, other: "Hex") -> int:
        return max(abs(self.q - other.q), abs(self.r - other.r), abs(self.s - other.s))

    def neighbors(self) -> tuple["Hex", ...]:
        directions = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
        return tuple(Hex(self.q + q, self.r + r) for q, r in directions)


@dataclass
class Unit:
    id: str
    faction: Faction
    position: Hex
    hp: float = 100.0
    attack: float = 25.0
    attack_range: int = 2
    movement: float = 2.0
    sensor_range: int = 4
    ew_power: float = 0.5
    jammed_for: float = 0.0
    cooldown: float = 0.0
    controller: str = "human"

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["faction"] = self.faction.value
        value["position"] = asdict(self.position)
        return value


@dataclass(frozen=True)
class Action:
    actor_id: str
    kind: str
    target_hex: Hex | None = None
    target_id: str | None = None


@dataclass
class Event:
    time: float
    kind: str
    actor_id: str
    target_id: str | None = None
    detail: str = ""
