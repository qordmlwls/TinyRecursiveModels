"""Offline planner utilities for Tiny Recursive Models.

This module implements a light-weight beam search that reuses the standard
TRM ACT loop without introducing any new learnable parameters.  It is meant
to be used in offline data generation or analysis pipelines where we want to
explore multiple ACT steps per puzzle under ``torch.no_grad()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import torch

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy


@dataclass
class PlannerNode:
    """Represents a single node in the ACT-level search tree."""

    carry: TinyRecursiveReasoningModel_ACTV1Carry
    score: float
    step: int
    outputs: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TRMPlannerConfig:
    """Configuration knobs for the offline planner."""

    beam_size: int = 4
    max_steps: int = 16
    max_nodes_per_beam: int = 64
    score_alpha: float = 1.0  # weight for label-based quality score
    score_beta: float = 0.0  # weight for q_halt confidence
    step_penalty: float = 0.01
    noise_std: float = 0.0  # std-dev of Gaussian noise for diverse trajectories


class TRMPlanner:
    """Simple beam-search planner that expands ACT steps under no-grad."""

    def __init__(
        self,
        model: TinyRecursiveReasoningModel_ACTV1,
        config: Optional[TRMPlannerConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model.eval()
        self.config = config or TRMPlannerConfig()
        self.device = device or next(self.model.parameters()).device

    @torch.no_grad()
    def plan_batch(self, batch: Dict[str, torch.Tensor]) -> List[PlannerNode]:
        """Runs beam search independently for every puzzle in ``batch``."""

        batch = self._move_to_device(batch, self.device)
        batch_size = batch["inputs"].shape[0]
        best_nodes: List[PlannerNode] = []

        for sample_idx in range(batch_size):
            puzzle_batch = self._select_example(batch, sample_idx)
            beam = self._initialize_beam(puzzle_batch)
            for _ in range(self.config.max_steps):
                beam = self._advance_beam(beam, puzzle_batch)
            best_nodes.append(max(beam, key=lambda node: node.score))

        return best_nodes

    # ------------------------------------------------------------------
    # Beam helpers

    def _initialize_beam(self, batch: Dict[str, torch.Tensor]) -> List[PlannerNode]:
        carry = self.model.initial_carry(batch)
        return [PlannerNode(carry=carry, score=0.0, step=0, outputs=None)]

    def _advance_beam(
        self,
        beam: List[PlannerNode],
        batch: Dict[str, torch.Tensor],
    ) -> List[PlannerNode]:
        children: List[PlannerNode] = []
        for node in beam:
            if node.metadata.get("halted", False):
                children.append(node)
            else:
                children.extend(self._expand_node(node, batch))

        if not children:
            return beam

        children.sort(key=lambda n: n.score, reverse=True)
        top_k = min(self.config.beam_size, len(children))
        return children[:top_k]

    def _expand_node(
        self,
        node: PlannerNode,
        batch: Dict[str, torch.Tensor],
    ) -> Sequence[PlannerNode]:
        new_carry, outputs = self._forward_once(node.carry, batch)
        step = node.step + 1
        score_continue = self._score_candidate(outputs, step, batch=batch)

        continue_node = PlannerNode(
            carry=new_carry,
            score=score_continue,
            step=step,
            outputs=outputs,
            metadata={"parent_score": node.score, "halted": False},
        )

        halt_node = PlannerNode(
            carry=new_carry,
            score=score_continue,
            step=step,
            outputs=outputs,
            metadata={"parent_score": node.score, "halted": True},
        )
        return [continue_node, halt_node]

    # ------------------------------------------------------------------
    # Low-level utilities

    def _forward_once(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        carry = self._move_carry_to_device(carry, self.device)
        new_carry, outputs = self.model(carry, batch)

        if self.config.noise_std > 0:
            new_carry = self._inject_noise_into_carry(new_carry, self.config.noise_std)
        # Ignore the ACT halting flags so search can continue for all nodes.
        new_carry = TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=new_carry.inner_carry,
            steps=new_carry.steps,
            halted=torch.zeros_like(new_carry.halted, dtype=torch.bool),
            current_data=new_carry.current_data,
        )
        return new_carry, outputs

    def _inject_noise_into_carry(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        noise_std: float,
    ) -> TinyRecursiveReasoningModel_ACTV1Carry:
        """Adds small Gaussian noise to the inner carry states."""

        inner = carry.inner_carry
        if hasattr(inner, "z_L") and getattr(inner, "z_L") is not None:
            z_L = getattr(inner, "z_L")
            inner = replace(inner, z_L=z_L + torch.randn_like(z_L) * noise_std)
        if hasattr(inner, "z_H") and getattr(inner, "z_H") is not None:
            z_H = getattr(inner, "z_H")
            inner = replace(inner, z_H=z_H + torch.randn_like(z_H) * noise_std)

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=inner,
            steps=carry.steps,
            halted=carry.halted,
            current_data=carry.current_data,
        )

    def _move_carry_to_device(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        device: torch.device,
    ) -> TinyRecursiveReasoningModel_ACTV1Carry:
        inner = carry.inner_carry
        z_H = inner.z_H.to(device)
        z_L = inner.z_L.to(device)
        inner = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)

        steps = carry.steps.to(device)
        halted = carry.halted.to(device)
        current_data = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in carry.current_data.items()
        }

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=inner,
            steps=steps,
            halted=halted,
            current_data=current_data,
        )

    def _score_candidate(
        self,
        outputs: Dict[str, torch.Tensor],
        step: int,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        """Combine label-aware quality with q_halt confidence."""

        labels = batch.get("labels")
        quality = 0.0
        if labels is not None:
            mask = (labels != IGNORE_LABEL_ID)
            ce = stablemax_cross_entropy(
                outputs["logits"],
                labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=mask,
            )
            denom = mask.sum().clamp_min(1)
            per_example_loss = (ce.sum() / denom).item()
            quality = math.exp(-per_example_loss)

        q_halt_logits = outputs["q_halt_logits"]
        if isinstance(q_halt_logits, tuple):
            q_halt_logits = q_halt_logits[0]
        halt_score = torch.sigmoid(q_halt_logits).mean().item()
        penalty = self.config.step_penalty * step
        return (
            self.config.score_alpha * quality
            + self.config.score_beta * halt_score
            - penalty
        )

    # ------------------------------------------------------------------
    # Batch utilities

    @staticmethod
    def _move_to_device(
        batch: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @staticmethod
    def _select_example(
        batch: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, torch.Tensor]:
        return {k: v[index : index + 1].clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
