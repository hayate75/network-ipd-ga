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

    output_dir: Path
    output_base: str

    def as_dict(self) -> dict:
        """ログ出力や保存用に辞書に変換"""
        return {
            "num_agents": self.num_agents,
            "generations": self.generations,
            "T": self.T,
            "mutation_rate": self.mutation_rate,
            "topology": self.topology,
            "small_world_k": self.small_world_k,
            "small_world_p": self.small_world_p,
            "scale_free_m": self.scale_free_m,
            "meta_influence": self.meta_influence,
            "output_dir": str(self.output_dir),
            "output_base": self.output_base,
        }


def load_config(path: Path | None = None) -> SimulationConfig:
    """YAML 設定ファイルを読み込み、SimulationConfig にして返す。"""
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
        output_dir=Path(data["output_dir"]),
        output_base=data["output_base"],
    )
