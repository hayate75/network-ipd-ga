# metrics.py
from __future__ import annotations
from typing import List
from collections import Counter
import math

from network_ipd_ga.agent import Agent
from network_ipd_ga.strategy import cooperation_potential, Strategy


def strategy_diversity_entropy(agents: List[Agent]) -> float:
    """
    戦略分布のシャノンエントロピーを多様性指標として用いる。
    """
    counter = Counter(a.strategy for a in agents)
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


def cooperation_rate_from_strategies(agents: List[Agent]) -> float:
    """
    各戦略の協調ポテンシャルを平均した値を協力率の近似として用いる。
    （実際の対戦から計測したい場合は simulation 側でカウントする）
    """
    if not agents:
        return 0.0
    return sum(cooperation_potential(a.strategy) for a in agents) / len(agents)
