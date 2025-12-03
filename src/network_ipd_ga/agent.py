# agent.py
from __future__ import annotations
from dataclasses import dataclass
from network_ipd_ga.strategy import Strategy

@dataclass
class Agent:
    id: int
    strategy: Strategy
    payoff: float = 0.0

    def reset_payoff(self) -> None:
        self.payoff = 0.0
