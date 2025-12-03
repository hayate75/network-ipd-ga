# game.py
from __future__ import annotations
from typing import Tuple
import random

from network_ipd_ga.agent import Agent
from network_ipd_ga.strategy import decide_action, C, D

# ペイオフ表
PAYOFF_TABLE = {
    (C, C): (3, 3),
    (C, D): (0, 5),
    (D, C): (5, 0),
    (D, D): (1, 1),
}


def play_ipd(
    agent_i: Agent,
    agent_j: Agent,
    T: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """
    2エージェント間で T ラウンドの繰り返しIPDを実施し、
    両エージェントの payoff を更新する。

    戻り値:
        coop_actions: この対戦で出た協調(C)の総数（両者合計）
        total_actions: 行動総数（2 * T）
    """
    prev_i = None
    prev_j = None
    coop_actions = 0
    total_actions = 0

    for t in range(T):
        a_i = decide_action(agent_i.strategy, t, prev_j)
        a_j = decide_action(agent_j.strategy, t, prev_i)

        payoff_i, payoff_j = PAYOFF_TABLE[(a_i, a_j)]
        agent_i.payoff += payoff_i
        agent_j.payoff += payoff_j

        if a_i == C:
            coop_actions += 1
        if a_j == C:
            coop_actions += 1
        total_actions += 2

        prev_i, prev_j = a_i, a_j

    return coop_actions, total_actions
