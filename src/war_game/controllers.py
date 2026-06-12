"""Pluggable tactical controllers."""

from __future__ import annotations

import random
from typing import Protocol

from .engine import GameEngine
from .model import Action, Unit


class Controller(Protocol):
    def decide(self, unit: Unit, engine: GameEngine) -> Action: ...


class HeuristicController:
    """Simple opponent baseline: EW when useful, attack, otherwise close distance."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def decide(self, unit: Unit, engine: GameEngine) -> Action:
        enemies = [u for u in engine.visible_units(unit.faction) if u.faction != unit.faction]
        enemies = [u for u in enemies if u.attack > 0]
        if not enemies:
            return Action(unit.id, "wait")
        target = min(enemies, key=lambda e: unit.position.distance(e.position))
        distance = unit.position.distance(target.position)
        if distance <= unit.sensor_range and target.jammed_for <= 0 and self.rng.random() < 0.3:
            return Action(unit.id, "ew", target_id=target.id)
        if distance <= unit.attack_range:
            return Action(unit.id, "attack", target_id=target.id)
        occupied = {u.position for u in engine.units.values() if u.alive}
        options = [cell for cell in engine.map.neighbors(unit.position)
                   if engine.map.passable(cell) and cell not in occupied]
        destination = min(options, key=lambda cell: cell.distance(target.position), default=unit.position)
        return Action(unit.id, "move", target_hex=destination)
