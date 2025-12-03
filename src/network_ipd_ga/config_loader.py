from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class SimulationConfig:
    num_agents: int
    generations: int
    T: int

    mutation_rate: float

    topology: str

    small_world_k: int
    small_world_p: float
    scale_free_m: int

    meta_influence: float

    logs_dir: Path


def _find_default_config_path() -> Path:
    """
    このファイルの位置から親ディレクトリをたどり、
    どこかに config/config.yml があればそれを返す。
    （プロジェクトルートを想定）
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "config.yml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("config/config.yml が見つかりませんでした。")


def load_config(path: Path | None = None) -> SimulationConfig:
    """YAML 設定ファイルを読み込み、SimulationConfig にして返す。"""
    if path is None:
        path = _find_default_config_path()

    with path.open("r") as f:
        data = yaml.safe_load(f)

    return SimulationConfig(
        num_agents=data["num_agents"],
        generations=data["generations"],
        T=data["T"],
        mutation_rate=data["mutation_rate"],
        topology=data["topology"],
        small_world_k=data["small_world_k"],
        small_world_p=data["small_world_p"],
        scale_free_m=data["scale_free_m"],
        meta_influence=data["meta_influence"],
        logs_dir=Path(data["logs_dir"]),
    )
