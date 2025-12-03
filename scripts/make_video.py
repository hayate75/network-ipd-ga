#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from network_ipd_ga.config_loader import load_config  # ← 追加：config からパスを決める


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make an evolution video from IPD network simulation results."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (same as run_single_experiment).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed of the experiment to visualize.",
    )

    # 以下はオプション（指定しなければデフォルト）
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output (default: 150).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["spring", "circular", "kamada"],
        default="spring",
        help="Network layout type (default: spring).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(6, 6),
        help="Figure size (width height).",
    )
    return parser.parse_args()


def load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        G: nx.Graph = pickle.load(f)
    return G


def compute_layout(G: nx.Graph, layout: str):
    if layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return pos


def strategy_to_color(strategy_bits: str) -> str:
    """
    戦略ビット列 (例 '010') をノード色にマップする。
    適宜好きな色に変えてOK。
    """
    color_map = {
        "000": "black",
        "001": "tab:blue",
        "010": "tab:orange",
        "011": "tab:green",
        "100": "tab:red",
        "101": "tab:purple",
        "110": "tab:brown",
        "111": "tab:pink",
    }
    return color_map.get(strategy_bits, "gray")


def payoff_to_size(payoff: float) -> float:
    """
    利得をノードサイズにマップする。
    スケールは好みに応じて調整してください。
    """
    base = 100.0
    scale = 10.0
    return max(base, base + scale * payoff)


def main() -> None:
    args = parse_args()

    # --- config から各種パスを自動決定 ---
    cfg = load_config(Path(args.config))
    seed = args.seed

    # run_single_experiment.py で保存したファイル名と揃える想定：
    #   summary : <output_dir>/csvs/<output_base>_seed<seed>.csv
    #   nodes   : <output_dir>/csvs/<output_base>_seed<seed>_nodes.csv
    #   graph   : <output_dir>/pickles/<output_base>_seed<seed>_graph.pickle
    base = cfg.output_base
    out_dir: Path = cfg.output_dir  # config_loader が Path に変換している 

    graph_path = out_dir / "pickles" / f"{base}_seed{seed}_graph.pickle"
    nodes_path = out_dir / "csvs" / f"{base}_seed{seed}_nodes.csv"
    summary_path = out_dir / "csvs" / f"{base}_seed{seed}.csv"

    # 動画の出力先
    video_dir = out_dir / "videos"
    out_path = video_dir / f"{base}_seed{seed}.mp4"

    # --- データ読み込み ---
    print(f"[INFO] Loading graph from {graph_path} ...")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    G = load_graph(graph_path)

    print(f"[INFO] Loading node history from {nodes_path} ...")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Node history CSV not found: {nodes_path}")
    node_df = pd.read_csv(nodes_path)

    summary_df = None
    if summary_path.exists():
        print(f"[INFO] Loading summary from {summary_path} ...")
        summary_df = pd.read_csv(summary_path)
        summary_df.set_index("generation", inplace=True)
    else:
        print(f"[WARN] Summary CSV not found: {summary_path} (titles will be simpler)")

    generations = sorted(node_df["generation"].unique())
    print(f"[INFO] Generations: {generations[0]} .. {generations[-1]} "
          f"(total {len(generations)} frames)")

    # ノード順固定（レイアウトと色の順序を揃える）
    node_order = list(G.nodes())

    # レイアウト計算（固定）
    pos = compute_layout(G, args.layout)

    # --- 描画の準備 ---
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    plt.axis("off")

    scat = None

    def update(frame_idx: int):
        nonlocal scat
        gen = generations[frame_idx]
        ax.clear()
        ax.set_title("")
        plt.axis("off")

        # この世代のノード情報
        df_g = node_df[node_df["generation"] == gen]
        row_by_id = {int(row["node_id"]): row for _, row in df_g.iterrows()}

        colors = []
        sizes = []
        for nid in node_order:
            row = row_by_id.get(int(nid))
            if row is None:
                colors.append("gray")
                sizes.append(100.0)
            else:
                bits = str(row["strategy_bits"])
                payoff = float(row["payoff"])
                colors.append(strategy_to_color(bits))
                sizes.append(payoff_to_size(payoff))

        # 描画
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=1.0)
        scat = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=colors,
            node_size=sizes,
            ax=ax,
        )

        # タイトル（summary があればそこから情報を出す）
        if summary_df is not None and gen in summary_df.index:
            s = summary_df.loc[gen]
            title = (
                f"Generation {gen}\n"
                f"realized_coop_rate = {s['realized_coop_rate']:.3f}, "
                f"diversity = {s['diversity']:.3f}, "
                f"avg_payoff = {s['avg_payoff']:.3f}"
            )
        else:
            title = f"Generation {gen}"

        ax.set_title(title)
        return scat,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(generations),
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )

    # --- 動画として保存 ---
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".gif":
        print(f"[INFO] Saving GIF to {out_path} ...")
        writer = PillowWriter(fps=args.fps)
        ani.save(out_path, writer=writer, dpi=args.dpi)
    else:
        print(f"[INFO] Saving MP4 to {out_path} ...")
        writer = FFMpegWriter(fps=args.fps)
        ani.save(out_path, writer=writer, dpi=args.dpi)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
