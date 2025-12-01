# utils/beam.py
# Utilities for internalized beam search in TR²
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import math

import torch
import torch.nn.functional as F

@dataclass
class Node:
    z: torch.Tensor        # (1, Tz, Dz)
    q: torch.Tensor        # scalar prob in [0,1]
    u: torch.Tensor        # scalar uncertainty proxy >=0
    e: torch.Tensor        # (D,) diversity embedding (L2 or cosine space)
    p_halt: torch.Tensor   # scalar prob in [0,1]
    meta: Dict[str, Any]   # bookkeeping

@dataclass
class BeamConfig:
    beam_size: int = 2
    branch_factor: int = 2
    ucb_alpha: float = 0.4
    diversity_delta: float = 0.15
    max_budget_nodes: int = 64
    halt_bonus: float = 0.05

def gumbel_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sample Top-K indices via Gumbel-Softmax trick (no replacement)."""
    g = -torch.log(-torch.log(torch.rand_like(logits)))
    return (logits + g).topk(k, dim=-1).indices

def _pairwise_cosine(a: torch.Tensor) -> torch.Tensor:
    # a: (N, D)
    a = F.normalize(a, dim=-1)
    return a @ a.t()  # (N, N)

def select_diverse_topk(cands: List[Node], cfg: BeamConfig, stats: Dict[str, Any]) -> List[Node]:
    """Select beam using q + UCB and diversity constraint on embedding e."""
    if len(cands) <= cfg.beam_size:
        return cands

    # Build tensors
    device = cands[0].q.device if isinstance(cands[0].q, torch.Tensor) else torch.device("cpu")
    dtype = cands[0].q.dtype if isinstance(cands[0].q, torch.Tensor) else torch.float32

    scores = torch.stack([c.q.detach().to(device=device, dtype=dtype) for c in cands])
    uncerts = torch.stack([c.u.detach().to(device=device, dtype=dtype) for c in cands])

    tot = scores + cfg.ucb_alpha * uncerts

    # Sort by tot score
    order = torch.argsort(tot, descending=True).tolist()

    # Greedy diversity: take top, then skip near-duplicates by cosine
    selected: List[Node] = []
    selected_e = []
    for idx in order:
        cand = cands[idx]
        if len(selected) == 0:
            selected.append(cand); selected_e.append(cand.e.detach()); 
        else:
            E = torch.stack(selected_e, dim=0)  # (k, D)
            sim = F.cosine_similarity(E, cand.e.detach().unsqueeze(0), dim=-1)  # (k,)
            if torch.all(sim < (1.0 - cfg.diversity_delta)):
                selected.append(cand); selected_e.append(cand.e.detach())
        if len(selected) >= cfg.beam_size:
            break
    # If not enough due to strict diversity, fill remaining by tot
    if len(selected) < cfg.beam_size:
        for idx in order:
            if not any(existing is cands[idx] for existing in selected):
                selected.append(cands[idx])
                if len(selected) >= cfg.beam_size:
                    break
    return selected
