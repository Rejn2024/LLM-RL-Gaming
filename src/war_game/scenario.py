"""Scenarios and procedural terrain generation."""

from __future__ import annotations

import random

from .map import HexMap
from .model import Faction, Hex, Terrain, Unit


def demo_scenario(seed: int = 7) -> tuple[HexMap, list[Unit]]:
    rng = random.Random(seed)
    weights = [Terrain.OPEN] * 11 + [Terrain.FOREST] * 3 + [Terrain.HILL] * 2 + [Terrain.URBAN]
    terrain = {Hex(q, r): rng.choice(weights) for q in range(12) for r in range(9)}
    terrain.update({Hex(5, r): Terrain.WATER for r in range(2, 7)})
    terrain[Hex(5, 4)] = Terrain.OPEN
    game_map = HexMap(12, 9, terrain)
    units = [
        Unit("blue-1", Faction.BLUE, Hex(1, 2), controller="human", ew_power=0.75),
        Unit("blue-2", Faction.BLUE, Hex(1, 6), controller="rl"),
        Unit("red-1", Faction.RED, Hex(10, 2), controller="bot", ew_power=0.65),
        Unit("red-2", Faction.RED, Hex(10, 6), controller="bot"),
        Unit("white-1", Faction.WHITE, Hex(6, 7), attack=0, controller="observer"),
    ]
    return game_map, units
