from __future__ import annotations
from dataclasses import dataclass
import math
import os

@dataclass
class CacheMLConfig:
   
    acf_threshold: float = 0.05
    acf_max_lag: int = 4096
    top_hot_objects: int = 50
    latency_budget_us: float = 1000.0
    success_prob: float = 0.99
    ram_fraction: float = 0.85          
    reserve_ram_mb: float = 1024.0      
    workers: int = 0                   
    random_state: int = 42
    min_trees: int = 32
    max_trees: int = 1024
    min_depth: int = 2
    max_depth: int = 16
    imbalance_strategy: str = "hybrid"
    downsample_max_per_object: int = 0     
    sample_weight_max: float = 20.0
    hot_weight_strength: float = 0.50
    rare_weight_strength: float = 0.75
    size_weight_strength: float = 0.25
    max_rows: int = 0
    candidate_n: int = 0                # 0 = hardware-bound auto
    eviction_batch: int = 0             # 0 = hardware-bound auto

    @property
    def n_workers(self) -> int:
        if self.workers and self.workers > 0:
            return int(self.workers)
        return max(1, os.cpu_count() or 1)

    @staticmethod
    def p_from_nmax(n_max: int, P: float = 0.99) -> float:
        n_max = max(1, int(n_max))
        return float(1.0 - math.exp(math.log(1.0 - P) / n_max))
