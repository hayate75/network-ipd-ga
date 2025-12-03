# scripts/run_single_experiment.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pickle

from network_ipd_ga.config_loader import load_config
from network_ipd_ga.simulation import run_simulation

def setup_logging(log_dir: Path, output_base: str, seed: int):
    fname = f"{output_base}_seed{seed}.log"
    log_file = log_dir / "logs" / fname
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),                 # 標準出力
            logging.FileHandler(log_file, encoding="utf-8"),  # ログファイル
        ],
    )
    logging.info(f"Logging initialized → {log_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single IPD evolution experiment using a YAML config and a seed."
    )

    # 設定ファイルのパス
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    # 乱数シード（必須）
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for the simulation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 設定ファイル読み込み
    cfg = load_config(Path(args.config))

    # シミュレーション実行
    df, graph, agents, node_df = run_simulation(
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

    # メイン出力ファイル名決定（世代サマリ）
    summary_fname = f"{cfg.output_base}_seed{args.seed}.csv"
    summary_path = cfg.output_dir / "csvs" / summary_fname

    # ノード履歴とグラフのファイル名
    node_fname = f"{cfg.output_base}_seed{args.seed}_nodes.csv"
    node_path = cfg.output_dir / "csvs" / node_fname

    graph_fname = f"{cfg.output_base}_seed{args.seed}_graph.pickle"
    graph_path = cfg.output_dir / "pickles" / graph_fname

    # 親ディレクトリ作成
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    node_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV 保存（世代サマリ）
    df.to_csv(summary_path, index=False)

    # CSV 保存（ノード履歴）
    node_df.to_csv(node_path, index=False)

    # グラフ保存（ネットワーク構造）
    with graph_path.open("wb") as f:
        pickle.dump(graph, f)

    logging.info(f"Saved summary to: {summary_path.resolve()}")
    logging.info(f"Saved node history to: {node_path.resolve()}")
    logging.info(f"Saved graph to: {graph_path.resolve()}")

    print(df.head())


if __name__ == "__main__":
    # ロギング初期化
    args = parse_args()
    cfg = load_config(Path(args.config))
    setup_logging(cfg.output_dir, cfg.output_base, args.seed)
    logger = logging.getLogger(__name__)
    logger.info("===== Simulation Config =====")
    for k, v in cfg.as_dict().items():
        logger.info(f"{k}: {v}")
    logger.info("=================================")

    main()
