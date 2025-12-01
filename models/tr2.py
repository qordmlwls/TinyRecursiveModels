from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)
from models.layers import CastedLinear
from utils.beam import BeamConfig, Node, gumbel_topk, select_diverse_topk


@dataclass
class TR2Extras:
    beam_size: int = 2
    branch_factor: int = 2
    ucb_alpha: float = 0.4
    diversity_delta: float = 0.15
    max_budget_nodes: int = 64
    halt_threshold: float = 0.5
    use_gumbel_branch: bool = True
    reinforce_branch: bool = False
    search_prob: float = 1.0
    policy_moves: int = 16
    policy_weight: float = 0.1
    inter_weight: float = 0.5
    div_weight: float = 0.01
    value_weight: float = 0.2
    halt_bonus: float = 0.05
    L_eval_cycles: int = 1


class TreeRecursiveReasoningModelInner(TinyRecursiveReasoningModel_ACTV1_Inner):
    """Inner reasoning module with lightweight beam-search branching."""

    def __init__(self, config, extras: TR2Extras):
        super().__init__(config)
        self.extras = extras
        self._force_enable_search: Optional[bool] = None

        self.state_dim = self.config.hidden_size * 2
        self.policy_head = CastedLinear(self.state_dim, self.extras.policy_moves, bias=True)
        self.move_embeddings = nn.Embedding(self.extras.policy_moves, self.config.hidden_size)
        nn.init.normal_(self.move_embeddings.weight, mean=0.0, std=1e-3)

        self.diversity_proj = CastedLinear(self.state_dim, self.config.hidden_size, bias=False)
        self.value_head = CastedLinear(self.state_dim, 2, bias=True)
        self.beam_cfg = BeamConfig(
            beam_size=self.extras.beam_size,
            branch_factor=self.extras.branch_factor,
            ucb_alpha=self.extras.ucb_alpha,
            diversity_delta=self.extras.diversity_delta,
            max_budget_nodes=self.extras.max_budget_nodes,
            halt_bonus=self.extras.halt_bonus,
        )
        max_reasonable_budget = max(
            self.beam_cfg.beam_size,
            self.beam_cfg.beam_size * max(1, self.beam_cfg.branch_factor + 1) * max(1, self.config.H_cycles),
        )
        if self.beam_cfg.max_budget_nodes <= 0 or self.beam_cfg.max_budget_nodes > max_reasonable_budget:
            self.beam_cfg.max_budget_nodes = max_reasonable_budget
        self._last_aux_losses: Dict[str, torch.Tensor] = {}
        self._search_rng = torch.Generator(device="cpu")

    def pop_aux_losses(self) -> Dict[str, torch.Tensor]:
        aux = self._last_aux_losses
        self._last_aux_losses = {}
        return aux

    def forward(  # type: ignore[override]
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        inputs = batch["inputs"]
        puzzle_ids = batch["puzzle_identifiers"]
        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID

        self._last_aux_losses = {}

        input_embeddings = self._input_embeddings(inputs, puzzle_ids)
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)

        batch_size = inputs.shape[0]

        beams: List[List[Node]] = []
        policy_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        diversity_terms: List[torch.Tensor] = []
        inter_step_losses: List[torch.Tensor] = []
        if self._force_enable_search is not None:
            enable_search = self._force_enable_search
        elif self.training and self.extras.search_prob < 1.0:
            enable_search = self._sample_search_toggle(carry.z_H.device)
        else:
            enable_search = True

        for b in range(batch_size):
            z_H0 = carry.z_H[b : b + 1]
            z_L0 = carry.z_L[b : b + 1]
            state_repr = self._state_repr(
                TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H0, z_L=z_L0),
                input_embeddings[b : b + 1],
            )
            div_vec = F.normalize(self.diversity_proj(state_repr), dim=-1).squeeze(0)
            with torch.no_grad():
                q_logits0 = self.q_head(z_H0[:, 0]).to(torch.float32)
            p0 = torch.sigmoid(q_logits0[..., 0] - q_logits0[..., 1]).squeeze(0)
            init_node = Node(
                z=z_H0.detach(),
                q=torch.tensor(0.0, device=z_H0.device),
                u=torch.tensor(0.0, device=z_H0.device),
                e=div_vec.detach(),
                p_halt=p0.detach(),
                meta={
                    "moves": [],
                    "z_L": z_L0.detach(),
                },
            )
            beams.append([init_node])

        nodes_used = [0] * batch_size
        budget = self.beam_cfg.max_budget_nodes if self.beam_cfg.max_budget_nodes > 0 else None
        for step in range(self.config.H_cycles):
            if budget is not None and all(n >= budget for n in nodes_used):
                break

            # First pass: decide which nodes to expand per sample
            keep_nodes_per_b: List[List[Node]] = [[] for _ in range(batch_size)]
            to_expand_per_b: List[List[Node]] = [[] for _ in range(batch_size)]
            for b in range(batch_size):
                if budget is not None and nodes_used[b] >= budget:
                    continue
                current_beam = beams[b]
                if not current_beam:
                    continue
                if not enable_search:
                    keep_nodes_per_b[b] = current_beam[: self.beam_cfg.beam_size]
                    continue
                p_halts = torch.stack([node.p_halt for node in current_beam]).to(torch.float32)
                expand_mask = p_halts <= self.extras.halt_threshold
                expand_idx = torch.nonzero(expand_mask, as_tuple=False).flatten().tolist()
                keep_idx = torch.nonzero(~expand_mask, as_tuple=False).flatten().tolist()
                keep_nodes_per_b[b] = [current_beam[i] for i in keep_idx]
                to_expand_per_b[b] = [current_beam[i] for i in expand_idx]

            # Build combined expansion batch over all samples
            parents_all: List[Node] = []
            sample_ids: List[int] = []
            for b in range(batch_size):
                for node in to_expand_per_b[b]:
                    parents_all.append(node)
                    sample_ids.append(b)

            children_all: List[List[Node]] = []
            if parents_all:
                parent_z_H = torch.cat([node.z for node in parents_all], dim=0)
                parent_z_L = torch.cat([node.meta["z_L"] for node in parents_all], dim=0)
                parent_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=parent_z_H, z_L=parent_z_L)
                sample_idx = torch.tensor(sample_ids, device=input_embeddings.device, dtype=torch.long)
                combined_inputs = input_embeddings.index_select(0, sample_idx)

                children_all = self._expand_nodes(
                    parents_all,
                    combined_inputs,
                    seq_info,
                )

            # Distribute children back per sample and select next beams
            child_index = 0
            for b in range(batch_size):
                current_beam = beams[b]
                keep_nodes = keep_nodes_per_b[b]
                to_expand = to_expand_per_b[b]
                new_children: List[Node] = []
                if to_expand:
                    num_parents = len(to_expand)
                    if children_all:
                        for _ in range(num_parents):
                            parent_children = children_all[child_index]
                            child_index += 1
                            if budget is not None:
                                remaining = budget - nodes_used[b]
                                if remaining <= 0:
                                    continue
                                if remaining < len(parent_children):
                                    scores = torch.stack([child.q for child in parent_children])
                                    topk = torch.topk(scores, k=remaining, dim=0).indices.tolist()
                                    parent_children = [parent_children[i] for i in topk]
                            new_children.extend(parent_children)
                            nodes_used[b] += len(parent_children)

                expanded = keep_nodes + new_children
                if not expanded:
                    expanded = current_beam
                beams[b] = select_diverse_topk(expanded, self.beam_cfg, {"t": step, "b": b})

        initial_z_H: List[torch.Tensor] = []
        initial_z_L: List[torch.Tensor] = []
        move_sequences: List[List[int]] = []
        for b in range(batch_size):
            initial_z_H.append(carry.z_H[b : b + 1])
            initial_z_L.append(carry.z_L[b : b + 1])
            beam_scores = torch.stack([n.q for n in beams[b]])
            best_idx = int(torch.argmax(beam_scores).item())
            best_node = beams[b][best_idx]
            moves: List[int] = list(best_node.meta["moves"])
            if len(moves) == 0:
                moves = [-1]
            move_sequences.append(moves)

        lengths = torch.tensor([len(seq) for seq in move_sequences], device=carry.z_H.device, dtype=torch.long)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 1
        moves_tensor = torch.full(
            (batch_size, max_len),
            -1,
            device=carry.z_H.device,
            dtype=torch.long,
        )
        for b, seq in enumerate(move_sequences):
            moves_tensor[b, : len(seq)] = torch.tensor(seq, device=carry.z_H.device, dtype=torch.long)

        initial_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.cat(initial_z_H, dim=0),
            z_L=torch.cat(initial_z_L, dim=0),
        )
        (
            combined_carry,
            logits_out,
            q_logits_out,
            prefix_logits_per_sample,
            prefix_e_per_sample,
        ) = self._run_sequences_batched(
            initial_carry,
            input_embeddings,
            seq_info,
            moves_tensor,
            lengths,
            labels,
            mask,
            policy_losses,
            value_losses,
        )

        q_halt_out = q_logits_out[..., 0]
        q_cont_out = q_logits_out[..., 1]

        for b in range(batch_size):
            if self.training and self.extras.inter_weight > 0:
                prefixes = prefix_logits_per_sample[b]
                if len(prefixes) > 1:
                    for Ls in prefixes[:-1]:
                        inter_loss = stablemax_cross_entropy(
                            Ls,
                            labels[b : b + 1],
                            ignore_index=IGNORE_LABEL_ID,
                            valid_mask=mask[b : b + 1],
                        ).sum() / mask[b : b + 1].sum().clamp_min(1)
                        inter_step_losses.append(inter_loss)
            if self.training and self.extras.div_weight > 0:
                prefix_e = prefix_e_per_sample[b]
                if len(prefix_e) > 1:
                    e_stack = torch.cat(prefix_e, dim=0)
                    if e_stack.size(0) > 1:
                        sims = F.cosine_similarity(
                            e_stack[-1:].expand(e_stack.size(0) - 1, -1),
                            e_stack[:-1],
                            dim=-1,
                        )
                        diversity_terms.append(sims.mean())

        aux_losses: Dict[str, torch.Tensor] = {}
        if inter_step_losses:
            aux_losses["inter"] = torch.stack(inter_step_losses).mean()
        if diversity_terms:
            aux_losses["diversity"] = torch.stack(diversity_terms).mean()
        if policy_losses:
            aux_losses["policy"] = torch.stack(policy_losses).mean()
        if value_losses:
            aux_losses["value"] = torch.stack(value_losses).mean()
        self._last_aux_losses = aux_losses

        self._force_enable_search = None
        return combined_carry, logits_out, (q_halt_out, q_cont_out)

    def set_force_enable_search(self, flag: Optional[bool]) -> None:
        self._force_enable_search = flag

    def _sample_search_toggle(self, device: torch.device) -> bool:
        """Sample (and synchronize) whether to enable the search phase."""
        threshold = torch.rand((), device=device)
        decision = threshold <= self.extras.search_prob
        if dist.is_available() and dist.is_initialized():
            tensor_flag = decision.to(torch.float32).clone()
            dist.broadcast(tensor_flag, src=0)
            return bool(tensor_flag.item() > 0.5)
        return bool(decision.item())

    def _expand_nodes(
        self,
        parents: List[Node],
        parent_inputs: torch.Tensor,
        seq_info: Dict[str, Any],
    ) -> List[List[Node]]:
        if not parents:
            return []

        device = parent_inputs.device
        num_parents = len(parents)

        parent_z_H = torch.cat([node.z for node in parents], dim=0)
        parent_z_L = torch.cat([node.meta["z_L"] for node in parents], dim=0)
        parent_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=parent_z_H,
            z_L=parent_z_L,
        )
        expanded_inputs = parent_inputs
        state_repr = self._state_repr(parent_carry, expanded_inputs)
        policy_logits = self.policy_head(state_repr)

        branch_k = min(self.beam_cfg.branch_factor, policy_logits.shape[-1])
        if branch_k > 0:
            if self.training and self.extras.use_gumbel_branch:
                move_idx = gumbel_topk(policy_logits, branch_k)
            else:
                move_idx = policy_logits.topk(branch_k, dim=-1).indices
        else:
            move_idx = torch.empty(num_parents, 0, device=device, dtype=torch.long)

        moves_per_parent = 1 + move_idx.shape[-1]
        moves_matrix = torch.full(
            (num_parents, moves_per_parent),
            -1,
            device=device,
            dtype=torch.long,
        )
        if branch_k > 0:
            moves_matrix[:, 1:] = move_idx

        parent_indices = torch.arange(num_parents, device=device, dtype=torch.long).repeat_interleave(moves_per_parent)
        flat_moves = moves_matrix.reshape(-1)
        candidate_z_H = parent_z_H.repeat_interleave(moves_per_parent, dim=0)
        candidate_z_L = parent_z_L.repeat_interleave(moves_per_parent, dim=0)
        candidate_inputs = expanded_inputs.repeat_interleave(moves_per_parent, dim=0)
        next_carry, q_hat, u_hat, halt_prob, div_vec = self._simulate_step_eval_batch(
            TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=candidate_z_H, z_L=candidate_z_L),
            candidate_inputs,
            seq_info,
            flat_moves,
        )

        scores_tensor = (q_hat + self.beam_cfg.halt_bonus * halt_prob).detach()

        # Shortlist per parent: keep NOOP (col 0) and top-M by score
        scores_parent = scores_tensor.view(num_parents, moves_per_parent)
        keep_k = min(self.beam_cfg.branch_factor, moves_per_parent)
        _, top_idx = torch.topk(scores_parent, k=keep_k, dim=1)
        keep_mask = torch.zeros_like(scores_parent, dtype=torch.bool)
        keep_mask.scatter_(1, top_idx, True)
        keep_mask[:, 0] = True  # always keep NOOP
        keep_mask_flat = keep_mask.view(-1)

        sel_indices = torch.nonzero(keep_mask_flat, as_tuple=False).flatten()
        sel_list = sel_indices.tolist()
        parent_indices_list = parent_indices.tolist()
        flat_moves_list = flat_moves.tolist()

        parent_moves_cache = [list(node.meta["moves"]) for node in parents]
        grouped_children: List[List[Node]] = [[] for _ in range(num_parents)]

        for idx in sel_list:
            parent_idx = parent_indices_list[idx]
            move_id = flat_moves_list[idx]
            moves_trace = list(parent_moves_cache[parent_idx])
            moves_trace.append(move_id)

            final_score = scores_tensor[idx]
            grouped_children[parent_idx].append(
                Node(
                    z=next_carry.z_H[idx : idx + 1].detach(),
                    q=final_score.to(device=device, dtype=torch.float32),
                    u=u_hat[idx].detach().to(device, dtype=torch.float32),
                    e=div_vec[idx].detach().to(device, dtype=torch.float32),
                    p_halt=halt_prob[idx].detach().to(device),
                    meta={
                        "moves": moves_trace,
                        "z_L": next_carry.z_L[idx : idx + 1].detach(),
                    },
                )
            )

        return grouped_children

    def _simulate_step_eval_batch(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Any],
        move_ids: torch.Tensor,
    ) -> tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_H = carry.z_H.clone()
        z_L = carry.z_L.clone()

        move_ids = move_ids.to(torch.long)
        if torch.any(move_ids >= 0):
            valid_mask = move_ids >= 0
            deltas = self.move_embeddings.weight[move_ids[valid_mask]].to(z_H.dtype)
            z_H = z_H.clone()
            z_H[valid_mask, 0, :] = z_H[valid_mask, 0, :] + deltas

        with torch.no_grad():
            steps = max(1, min(self.extras.L_eval_cycles, self.config.L_cycles))
            for _ in range(steps):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
            state_repr = self._state_repr_from_states(z_H, input_embeddings)
            div_vec = F.normalize(self.diversity_proj(state_repr), dim=-1)
            value_logits = self.value_head(state_repr).to(torch.float32)
            q_hat = torch.sigmoid(value_logits[..., 0])
            u_hat = F.softplus(value_logits[..., 1])
            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            halt_prob = torch.sigmoid(q_logits[..., 0] - q_logits[..., 1])

        next_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        return next_carry, q_hat.detach(), u_hat.detach(), halt_prob.detach(), div_vec.detach()

    def _run_sequences_batched(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Any],
        moves: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        policy_losses: List[torch.Tensor],
        value_losses: List[torch.Tensor],
    ) -> tuple[
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        torch.Tensor,
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        batch_size = moves.size(0)
        z_H = carry.z_H
        z_L = carry.z_L

        prefix_logits: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        prefix_e: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        logits_final: List[Optional[torch.Tensor]] = [None] * batch_size
        q_logits_final: List[Optional[torch.Tensor]] = [None] * batch_size

        max_len = moves.size(1)
        for t in range(max_len):
            active_mask = lengths > t
            if not torch.any(active_mask):
                break
            active_idx = torch.nonzero(active_mask, as_tuple=False).flatten()
            step_moves = moves.index_select(0, active_idx)[:, t]
            sub_z_H = z_H.index_select(0, active_idx)
            sub_z_L = z_L.index_select(0, active_idx)
            sub_inputs = input_embeddings.index_select(0, active_idx)
            sub_labels = labels.index_select(0, active_idx)
            sub_label_mask = label_mask.index_select(0, active_idx)

            sub_z_H_new, sub_z_L_new, sub_logits, sub_q_logits, sub_e = self._simulate_step_train_batched(
                sub_z_H,
                sub_z_L,
                sub_inputs,
                seq_info,
                step_moves,
                sub_labels,
                sub_label_mask,
                policy_losses,
                value_losses,
            )

            z_H = z_H.clone()
            z_L = z_L.clone()
            z_H.index_copy_(0, active_idx, sub_z_H_new)
            z_L.index_copy_(0, active_idx, sub_z_L_new)

            active_list = active_idx.tolist()
            for j, b_idx in enumerate(active_list):
                logits_step = sub_logits[j : j + 1]
                e_step = sub_e[j : j + 1]
                prefix_logits[b_idx].append(logits_step)
                prefix_e[b_idx].append(e_step)
                logits_final[b_idx] = logits_step
                q_logits_final[b_idx] = sub_q_logits[j : j + 1]

        final_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        logits_out = torch.cat([tensor for tensor in logits_final if tensor is not None], dim=0)
        q_logits_out = torch.cat([tensor for tensor in q_logits_final if tensor is not None], dim=0)
        return final_carry, logits_out, q_logits_out, prefix_logits, prefix_e

    def _simulate_step_train_batched(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Any],
        move_ids: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        policy_losses: List[torch.Tensor],
        value_losses: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_dtype = z_H.dtype
        state_repr = self._state_repr_from_states(z_H, input_embeddings).to(torch.float32)

        valid_mask = move_ids >= 0
        if valid_mask.any():
            if self.training:
                logits = self.policy_head(state_repr[valid_mask])
                probs = F.softmax(logits, dim=-1, dtype=torch.float32)
                chosen = move_ids[valid_mask]
                one_hot = F.one_hot(chosen, num_classes=logits.size(-1)).to(torch.float32)
                st = (one_hot - probs.detach()) + probs
                delta_vec = torch.matmul(st, self.move_embeddings.weight.to(torch.float32))
                z_H_updates = z_H[valid_mask].clone().to(torch.float32)
                z_H_updates[:, 0, :] = z_H_updates[:, 0, :] + delta_vec
                z_H = z_H.clone()
                z_H[valid_mask] = z_H_updates.to(orig_dtype)

                if self.extras.policy_weight > 0:
                    policy_loss = F.cross_entropy(
                        logits.to(torch.float32),
                        chosen.to(torch.long),
                        reduction="mean",
                    )
                    policy_losses.append(policy_loss)
            else:
                deltas = self.move_embeddings.weight[move_ids[valid_mask]].to(orig_dtype)
                z_H = z_H.clone()
                z_H[valid_mask, 0, :] = z_H[valid_mask, 0, :] + deltas

        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        state_next = self._state_repr_from_states(z_H, input_embeddings)
        e_vec = F.normalize(self.diversity_proj(state_next), dim=-1)

        if self.training and self.extras.value_weight > 0:
            ce = stablemax_cross_entropy(
                logits.to(torch.float32),
                labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=label_mask,
            )
            token_counts = label_mask.sum(dim=1).clamp_min(1)
            per_example_loss = (ce.sum(dim=1) / token_counts).to(torch.float32)
            value_logits = self.value_head(state_next).to(torch.float32)
            pred_quality = torch.sigmoid(value_logits[..., 0])
            pred_uncert = F.softplus(value_logits[..., 1])
            value_target = torch.exp(-per_example_loss).detach()
            uncert_target = per_example_loss.detach()
            value_loss = F.mse_loss(pred_quality, value_target, reduction="mean") + F.mse_loss(
                pred_uncert, uncert_target, reduction="mean"
            )
            value_losses.append(value_loss)

        return z_H, z_L, logits, q_logits, e_vec

    def _run_sequence(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Any],
        moves: Sequence[int],
        return_prefix: bool = False,
    ) -> tuple[
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        if len(moves) == 0:
            moves = [-1]

        z_H = carry.z_H
        z_L = carry.z_L
        logits_out: Optional[torch.Tensor] = None
        q_logits_out: Optional[torch.Tensor] = None
        prefix_logits: List[torch.Tensor] = []
        prefix_e: List[torch.Tensor] = []

        for move_id in moves:
            z_H, z_L, logits_out, q_logits_out, e_vec = self._simulate_step_train(
                z_H,
                z_L,
                input_embeddings,
                seq_info,
                move_id,
            )
            if return_prefix:
                prefix_logits.append(logits_out)
                prefix_e.append(e_vec)

        assert logits_out is not None and q_logits_out is not None
        next_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        return next_carry, logits_out, (q_logits_out[..., 0], q_logits_out[..., 1]), prefix_logits, prefix_e

    def _simulate_step_train(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Any],
        move_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_dtype = z_H.dtype
        state_repr = self._state_repr_from_states(z_H, input_embeddings).to(torch.float32)
        if move_id >= 0:
            if self.training:
                z_H_float = z_H.clone().to(torch.float32)
                logits = self.policy_head(state_repr)
                probs = F.softmax(logits, dim=-1, dtype=torch.float32)
                one_hot = F.one_hot(
                    torch.tensor(move_id, device=logits.device, dtype=torch.long),
                    num_classes=logits.size(-1),
                ).to(torch.float32)
                st = (one_hot - probs.detach()) + probs
                delta_vec = torch.matmul(st, self.move_embeddings.weight.to(torch.float32))
                z_H_float[:, 0, :] = z_H_float[:, 0, :] + delta_vec
                z_H = z_H_float.to(orig_dtype)
            else:
                z_H = z_H.clone()
                delta = self.move_embeddings.weight[move_id].to(orig_dtype)
                z_H[:, 0, :] = z_H[:, 0, :] + delta

        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        logits = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        state_next = self._state_repr_from_states(z_H, input_embeddings)
        e_vec = F.normalize(self.diversity_proj(state_next), dim=-1)
        return z_H, z_L, logits, q_logits, e_vec

    def _state_repr(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return self._state_repr_from_states(carry.z_H, input_embeddings)

    def _state_repr_from_states(
        self,
        z_H: torch.Tensor,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        z_head = z_H[:, 0, :].to(torch.float32)
        x_pool = input_embeddings.mean(dim=1).to(torch.float32)
        return torch.cat([z_head, x_pool], dim=-1)


class TR2(TinyRecursiveReasoningModel_ACTV1):
    """Tree-Recursive Model wrapper with internalised beam search."""

    def __init__(self, config_dict: Dict[str, Any]):
        extras_kwargs: Dict[str, Any] = {}
        base_config: Dict[str, Any] = {}
        for key, value in config_dict.items():
            if key in TR2Extras.__annotations__:
                extras_kwargs[key] = value
            else:
                base_config[key] = value

        super().__init__(base_config)
        self.extras = TR2Extras(**extras_kwargs)
        self.inner = TreeRecursiveReasoningModelInner(self.config, self.extras)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_ACTV1Carry:  # type: ignore[override]
        return super().initial_carry(batch)

    def forward(  # type: ignore[override]
        self,
        carry: TinyRecursiveReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_carry, outputs = super().forward(carry, batch)
        aux = self.inner.pop_aux_losses()
        if aux:
            total = torch.tensor(0.0, device=outputs["logits"].device, dtype=outputs["logits"].dtype)
            if "inter" in aux:
                total = total + self.extras.inter_weight * aux["inter"]
            if "diversity" in aux:
                total = total + self.extras.div_weight * aux["diversity"]
            if "policy" in aux:
                total = total + self.extras.policy_weight * aux["policy"]
            if "value" in aux:
                total = total + self.extras.value_weight * aux["value"]
            outputs["aux_loss"] = total
            outputs["aux_terms"] = {k: v.detach() for k, v in aux.items()}
        return new_carry, outputs

    def set_force_enable_search(self, flag: Optional[bool]) -> None:
        self.inner.set_force_enable_search(flag)
