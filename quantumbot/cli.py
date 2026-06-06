"""Command-line entry points."""

from __future__ import annotations

import argparse

from .simulation import Simulation, SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evolve bots controlled by a variational quantum circuit."
    )
    parser.add_argument("--headless", action="store_true", help="run without the GUI")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--steps", type=int, default=450, help="steps per generation")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generations < 1 or args.steps < 1:
        raise SystemExit("--generations and --steps must be positive")
    config = SimulationConfig(
        population_size=args.population,
        generation_steps=args.steps,
    )
    simulation = Simulation(config, seed=args.seed)

    if args.headless:
        for _ in range(args.generations):
            summary = simulation.run_generation()
            print(
                f"generation={summary.generation} "
                f"best={summary.best_score:.2f} "
                f"mean={summary.mean_score:.2f} "
                f"resources={summary.resources_collected} "
                f"mutation={summary.mutation_scale:.3f}"
            )
        return 0

    from .ui import QuantumBotApp

    QuantumBotApp(simulation).run()
    return 0
