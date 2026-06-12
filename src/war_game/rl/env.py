"""Compact RL environment for target search and EW decisions."""

from __future__ import annotations

import math
import random

import numpy as np

from ..engine import GameEngine
from ..map import HexMap
from ..model import Action, Faction, Hex, Unit


class EWSearchEnv:
    """Dependency-light Gym-like environment.

    Actions: 0-5 move to neighbor, 6 EW attack, 7 kinetic attack, 8 wait.
    Observation: normalized own/target state, relative position, range and jam status.
    """

    observation_size = 10
    action_size = 9

    def __init__(self, width: int = 9, height: int = 9, max_steps: int = 100, seed: int = 0):
        self.width, self.height, self.max_steps = width, height, max_steps
        self.rng = random.Random(seed)
        self.engine: GameEngine
        self.steps = 0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng.seed(seed)
        blue = Unit("blue-ew", Faction.BLUE, Hex(0, self.rng.randrange(self.height)), ew_power=0.8)
        red = Unit("red-target", Faction.RED, Hex(self.width - 1, self.rng.randrange(self.height)))
        self.engine = GameEngine(HexMap(self.width, self.height), [blue, red], self.rng.randrange(2**31))
        self.steps = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        own, target = self.engine.units["blue-ew"], self.engine.units["red-target"]
        dx = (target.position.q - own.position.q) / self.width
        dy = (target.position.r - own.position.r) / self.height
        distance = own.position.distance(target.position)
        visible = float(distance <= own.sensor_range)
        return np.array([own.position.q / self.width, own.position.r / self.height, dx, dy,
                         distance / max(self.width, self.height), visible, own.hp / 100,
                         target.hp / 100, min(target.jammed_for / 5, 1), own.cooldown], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        own, target = self.engine.units["blue-ew"], self.engine.units["red-target"]
        old_distance = own.position.distance(target.position)
        previous_jam = target.jammed_for
        if action < 6:
            destination = own.position.neighbors()[action]
            self.engine.submit(Action(own.id, "move", target_hex=destination))
        elif action == 6:
            self.engine.submit(Action(own.id, "ew", target_id=target.id))
        elif action == 7:
            self.engine.submit(Action(own.id, "attack", target_id=target.id))
        else:
            self.engine.submit(Action(own.id, "wait"))
        self.engine.step(1.5)
        self.steps += 1
        distance = own.position.distance(target.position)
        progress = 0.03 * math.copysign(1, old_distance - distance) if old_distance != distance else 0
        reward = -0.02 + progress
        if target.jammed_for > previous_jam:
            reward += 2.0
        if action == 6 and distance > own.sensor_range:
            reward -= 0.2
        if not target.alive:
            reward += 4.0
        terminated = not target.alive
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"distance": distance}
