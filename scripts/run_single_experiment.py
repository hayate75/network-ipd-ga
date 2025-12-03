# run_single_experiment.py
from __future__ import annotations
import argparse
from pathlib import Path

from network_ipd_ga.simulation import run_simulation, Topology, ModelType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single IPD meta-environment evolution experiment."
    )
    parser.add_argument("--topology", type=str, default="lattice",
                        choices=["lattice", "small_world", "scale_free"])
    parser.add_argument("--model", type=str, default="ga",
                        choices=["ga", "meta_ga"])
    parser.add_argument("--num_agents", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--T", type=int, default=50,
                        help="Number of IPD rounds per edge per generation.")
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--small_world_k", type=int, default=4)
    parser.add_argument("--small_world_p", type=float, default=0.1)
    parser.add_argument("--scale_free_m", type=int, default=2)
    parser.add_argument("--meta_influence", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="result_single_experiment.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df, graph, agents = run_simulation(
        topology=args.topology,          # type: ignore[arg-type]
        model_type=args.model,           # type: ignore[arg-type]
        num_agents=args.num_agents,
        generations=args.generations,
        T=args.T,
        mutation_rate=args.mutation_rate,
        small_world_k=args.small_world_k,
        small_world_p=args.small_world_p,
        scale_free_m=args.scale_free_m,
        seed=args.seed,
        meta_influence=args.meta_influence,
    )

    out_path = Path(args.output_csv)
    df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
