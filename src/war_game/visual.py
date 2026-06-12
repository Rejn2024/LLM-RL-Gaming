"""Optional pygame real-time visualizer."""

from __future__ import annotations

import math

from .controllers import HeuristicController
from .engine import GameEngine
from .model import Faction, Hex, Terrain

COLORS = {Faction.BLUE: (35, 110, 240), Faction.RED: (220, 55, 55), Faction.WHITE: (245, 245, 245)}
TERRAIN = {
    Terrain.OPEN: (155, 170, 115), Terrain.FOREST: (55, 105, 60),
    Terrain.URBAN: (110, 110, 110), Terrain.WATER: (45, 105, 170), Terrain.HILL: (145, 120, 75),
}


def center(cell: Hex, size: int = 36) -> tuple[float, float]:
    x = size * (math.sqrt(3) * cell.q + math.sqrt(3) / 2 * cell.r) + 55
    return x, size * 1.5 * cell.r + 55


def run(engine: GameEngine, fps: int = 30) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((900, 650))
    clock = pygame.time.Clock()
    pygame.display.set_caption("LLM + RL Hex War Game")
    bot, elapsed, running = HeuristicController(), 0.0, True
    while running:
        elapsed += clock.tick(fps) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if elapsed >= 0.5:
            for unit in engine.units.values():
                if unit.alive and unit.controller in {"bot", "rl"} and unit.cooldown <= 0:
                    engine.submit(bot.decide(unit, engine))
            engine.step(elapsed)
            elapsed = 0
        screen.fill((25, 28, 32))
        for cell, terrain in engine.map.terrain.items():
            x, y = center(cell)
            points = [
                (x + 36 * math.cos(math.radians(60 * i - 30)),
                 y + 36 * math.sin(math.radians(60 * i - 30))) for i in range(6)
            ]
            pygame.draw.polygon(screen, TERRAIN[terrain], points)
            pygame.draw.polygon(screen, (35, 35, 35), points, 2)
        for unit in engine.units.values():
            if not unit.alive:
                continue
            x, y = center(unit.position)
            pygame.draw.circle(screen, COLORS[unit.faction], (x, y), 16)
            pygame.draw.rect(screen, (20, 20, 20), (x - 20, y - 27, 40, 5))
            pygame.draw.rect(screen, (70, 230, 90), (x - 20, y - 27, 40 * unit.hp / 100, 5))
        pygame.display.flip()
    pygame.quit()
