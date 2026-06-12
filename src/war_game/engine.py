"""Real-time simulation engine with deterministic and stochastic modes."""

from __future__ import annotations

import random
from collections import deque

from .map import HexMap
from .model import Action, Event, Faction, Hex, TERRAIN_COST, TERRAIN_DEFENCE, Unit


class GameEngine:
    """Run game actions with fixed expected outcomes or sampled probabilistic outcomes.

    Deterministic mode is the default. Set ``stochastic=True`` at startup to sample
    attack damage and EW success using the supplied seed.
    """

    def __init__(
        self, game_map: HexMap, units: list[Unit], seed: int = 0, stochastic: bool = False
    ):
        self.map = game_map
        self.units = {unit.id: unit for unit in units}
        self.time = 0.0
        self.rng = random.Random(seed)
        self.stochastic = stochastic
        self.events: deque[Event] = deque(maxlen=500)
        self._queue: deque[Action] = deque()
        self._step_events: list[Event] = []

    def submit(self, action: Action) -> None:
        self._queue.append(action)

    def step(self, dt: float = 0.25) -> list[Event]:
        self._step_events = []
        for unit in self.units.values():
            unit.cooldown = max(0.0, unit.cooldown - dt)
            unit.jammed_for = max(0.0, unit.jammed_for - dt)
        while self._queue:
            self._apply(self._queue.popleft())
        self.time += dt
        return list(self._step_events)

    def _event(self, kind: str, actor: str, target: str | None = None, detail: str = ""):
        event = Event(self.time, kind, actor, target, detail)
        self.events.append(event)
        self._step_events.append(event)

    def _apply(self, action: Action) -> None:
        actor = self.units.get(action.actor_id)
        if not actor or not actor.alive or actor.cooldown > 0:
            return
        if action.kind == "move" and action.target_hex:
            self._move(actor, action.target_hex)
        elif action.kind == "attack" and action.target_id:
            self._attack(actor, action.target_id)
        elif action.kind == "ew" and action.target_id:
            self._ew(actor, action.target_id)
        elif action.kind == "wait":
            actor.cooldown = 0.5
            self._event("wait", actor.id)

    def _move(self, actor: Unit, destination: Hex) -> None:
        occupied = {u.position for u in self.units.values() if u.alive and u.id != actor.id}
        path = self.map.path(actor.position, destination, occupied)
        spent = 0.0
        reached = actor.position
        for cell in path:
            cost = TERRAIN_COST[self.map.terrain[cell]]
            if spent + cost > actor.movement:
                break
            reached, spent = cell, spent + cost
        if reached != actor.position:
            actor.position = reached
            actor.cooldown = max(0.25, spent * 0.4)
            self._event("move", actor.id, detail=f"to {reached.q},{reached.r}")

    def _valid_target(self, actor: Unit, target_id: str, max_range: int) -> Unit | None:
        target = self.units.get(target_id)
        if not target or not target.alive or target.faction == actor.faction:
            return None
        return target if actor.position.distance(target.position) <= max_range else None

    def _attack(self, actor: Unit, target_id: str) -> None:
        target = self._valid_target(actor, target_id, actor.attack_range)
        if not target:
            return
        defence = TERRAIN_DEFENCE[self.map.terrain[target.position]]
        jam_penalty = 0.55 if actor.jammed_for > 0 else 1.0
        damage_multiplier = self.rng.uniform(0.8, 1.2) if self.stochastic else 1.0
        damage = actor.attack * jam_penalty * (1 - defence) * damage_multiplier
        target.hp = max(0.0, target.hp - damage)
        actor.cooldown = 1.0
        self._event("attack", actor.id, target.id, f"damage={damage:.1f}")

    def _ew(self, actor: Unit, target_id: str) -> None:
        target = self._valid_target(actor, target_id, actor.sensor_range)
        if not target:
            return
        distance_penalty = actor.position.distance(target.position) / max(actor.sensor_range, 1)
        chance = max(0.1, min(0.9, actor.ew_power - 0.25 * distance_penalty))
        success = self.rng.random() < chance if self.stochastic else chance >= 0.5
        if success:
            target.jammed_for = max(target.jammed_for, 2.0 + actor.ew_power * 3.0)
        actor.cooldown = 1.5
        self._event("ew", actor.id, target.id, f"success={success};chance={chance:.2f}")

    def visible_units(self, faction: Faction) -> list[Unit]:
        observers = [u for u in self.units.values() if u.alive and u.faction == faction]
        return [
            u
            for u in self.units.values()
            if u.alive
            and (
                u.faction == faction
                or any(o.position.distance(u.position) <= o.sensor_range for o in observers)
            )
        ]

    def winner(self) -> Faction | None:
        combatants = {
            u.faction for u in self.units.values() if u.alive and u.faction != Faction.WHITE
        }
        return next(iter(combatants)) if len(combatants) == 1 else None
