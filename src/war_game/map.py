"""Hex map and pathfinding."""

from __future__ import annotations

import heapq
import itertools

from .model import Hex, TERRAIN_COST, Terrain


class HexMap:
    def __init__(self, width: int, height: int, terrain: dict[Hex, Terrain] | None = None):
        self.width = width
        self.height = height
        self.terrain = {cell: Terrain.OPEN for cell in self.cells()}
        self.terrain.update(terrain or {})

    def cells(self):
        for q in range(self.width):
            for r in range(self.height):
                yield Hex(q, r)

    def contains(self, cell: Hex) -> bool:
        return 0 <= cell.q < self.width and 0 <= cell.r < self.height

    def neighbors(self, cell: Hex):
        return (n for n in cell.neighbors() if self.contains(n))

    def passable(self, cell: Hex) -> bool:
        return self.contains(cell) and TERRAIN_COST[self.terrain[cell]] != float("inf")

    def path(self, start: Hex, goal: Hex, blocked: set[Hex] | None = None) -> list[Hex]:
        """A* path, excluding start and including goal; empty when unreachable."""
        blocked = blocked or set()
        frontier: list[tuple[float, int, Hex]] = [(0, 0, start)]
        serial = itertools.count()
        came_from: dict[Hex, Hex | None] = {start: None}
        cost = {start: 0.0}
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                result = []
                while current != start:
                    result.append(current)
                    current = came_from[current]  # type: ignore[assignment]
                return list(reversed(result))
            for nxt in self.neighbors(current):
                if nxt in blocked or not self.passable(nxt):
                    continue
                new_cost = cost[current] + TERRAIN_COST[self.terrain[nxt]]
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(frontier, (new_cost + nxt.distance(goal), next(serial), nxt))
        return []
