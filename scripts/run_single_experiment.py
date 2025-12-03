# run_single_experiment.py
from __future__ import annotations
import argparse
from pathlib import Path

from network_ipd_ga.simulation import run_simulation, Topology, ModelType
from network_ipd_ga.config_loader import load_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single IPD evolution experiment using a YAML config and a seed."
    )

    # 設定ファイルのパス
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yml",
        help="Path to the YAML configuration file.",
    )

    # 乱数シード
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for the simulation.",
    )

    # 出力ログファイル（結果CSV）
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV filename. If empty, auto-generate from config settings.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # 設定ファイル読み込み
    cfg = load_config(Path(args.config))

    # 実行
    df, graph, agents = run_simulation(
        topology=cfg.topology,
        num_agents=cfg.num_agents,
        generations=cfg.generations,
        T=cfg.T,
        mutation_rate=cfg.mutation_rate,
        small_world_k=cfg.small_world_k,
        small_world_p=cfg.small_world_p,
        scale_free_m=cfg.scale_free_m,
        meta_influence=cfg.meta_influence,
        seed=args.seed,
    )

    # 出力ファイル名を決定
    if args.output:
        out_path = Path(args.output)
    else:
        fname = f"{cfg.topology}_meta{cfg.meta_influence}_seed{args.seed}.csv"
        out_path = cfg.logs_dir / fname

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved results to {out_path}")
    print(df.head())
