"""Command-line entry point."""

import argparse

from .engine import GameEngine
from .scenario import demo_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-time hex war game")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    game_map, units = demo_scenario()
    engine = GameEngine(game_map, units)
    if args.headless:
        for _ in range(args.steps):
            engine.step()
        print(f"simulation_time={engine.time:.2f} units_alive={sum(u.alive for u in units)}")
    else:
        from .visual import run

        run(engine)


if __name__ == "__main__":
    main()
