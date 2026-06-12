"""Command-line entry point."""

import argparse

from .engine import GameEngine
from .scenario import demo_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-time hex war game")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="sample probabilistic outcomes instead of using deterministic expected outcomes",
    )
    args = parser.parse_args()
    game_map, units = demo_scenario(args.seed)
    engine = GameEngine(game_map, units, seed=args.seed, stochastic=args.stochastic)
    if args.headless:
        for _ in range(args.steps):
            engine.step()
        print(f"simulation_time={engine.time:.2f} units_alive={sum(u.alive for u in units)}")
    else:
        from .visual import run

        run(engine)


if __name__ == "__main__":
    main()
