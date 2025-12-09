from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Cast logits to float32 for stable loss computation in mixed precision
        logits_f = outputs["logits"].to(torch.float32)
        q_halt_logits_f = outputs["q_halt_logits"].to(torch.float32)
        q_continue_logits_f = outputs["q_continue_logits"].to(torch.float32) if "q_continue_logits" in outputs else None

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(logits_f, dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(logits_f, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(logits_f, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits_f,
            seq_is_correct.to(q_halt_logits_f.dtype),
            reduction="sum",
        )
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs and q_continue_logits_f is not None:
            target_q = outputs["target_q_continue"].to(torch.float32)
            q_continue_loss = F.binary_cross_entropy_with_logits(q_continue_logits_f, target_q, reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Keep accumulation in float32 for stable backward on mixed-precision models
        total_loss = (lm_loss + 0.5 * (q_halt_loss + q_continue_loss)).to(torch.float32)
        if "aux_loss" in outputs:
            aux = outputs["aux_loss"].to(torch.float32)
            total_loss = total_loss + aux
            metrics["aux_loss"] = aux.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class SearchDistillLossHead(ACTLossHead):
    def __init__(
        self,
        model: nn.Module,
        loss_type: str,
        teacher_weight: float = 0.5,
        teacher_step_weight: float = 0.0,
        kl_temperature: float = 0.0,
    ):
        super().__init__(model, loss_type)
        self.teacher_weight = teacher_weight
        self.teacher_step_weight = teacher_step_weight
        self.kl_temperature = kl_temperature

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        logits_f = outputs["logits"].to(torch.float32)
        q_halt_logits_f = outputs["q_halt_logits"].to(torch.float32)
        q_continue_logits_f = outputs["q_continue_logits"].to(torch.float32) if "q_continue_logits" in outputs else None

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits_f, dim=-1)

            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(logits_f, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        lm_loss = (self.loss_fn(logits_f, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits_f,
            seq_is_correct.to(q_halt_logits_f.dtype),
            reduction="sum",
        )
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        q_continue_loss = 0
        if "target_q_continue" in outputs and q_continue_logits_f is not None:
            target_q = outputs["target_q_continue"].to(torch.float32)
            q_continue_loss = F.binary_cross_entropy_with_logits(q_continue_logits_f, target_q, reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        total_loss = (lm_loss + 0.5 * (q_halt_loss + q_continue_loss)).to(torch.float32)

        if "aux_loss" in outputs:
            aux = outputs["aux_loss"].to(torch.float32)
            total_loss = total_loss + aux
            metrics["aux_loss"] = aux.detach()

        applied_distill = False
        teacher_tokens = new_carry.current_data.get("teacher_tokens")
        teacher_logits = new_carry.current_data.get("teacher_logits")

        if (
            teacher_logits is not None
            and self.kl_temperature > 0
            and self.teacher_weight > 0
        ):
            T = self.kl_temperature
            teacher_logits_f = teacher_logits.to(logits_f.dtype)
            student_logprobs = F.log_softmax(logits_f / T, dim=-1)
            teacher_probs = F.softmax(teacher_logits_f / T, dim=-1)
            kl_per_token = F.kl_div(
                student_logprobs, teacher_probs, reduction="none"
            ).sum(dim=-1)

            if teacher_tokens is not None:
                teacher_mask = (teacher_tokens != IGNORE_LABEL_ID)
                denom = teacher_mask.sum().clamp_min(1).to(kl_per_token.dtype)
                kl_loss = (kl_per_token * teacher_mask.to(kl_per_token.dtype)).sum() / denom
            else:
                kl_loss = kl_per_token.mean()

            total_loss = total_loss + self.teacher_weight * (T * T) * kl_loss
            metrics["distill_kl_loss"] = kl_loss.detach()
            applied_distill = True

        if (
            not applied_distill
            and self.teacher_weight > 0
            and teacher_tokens is not None
        ):
            teacher_mask = (teacher_tokens != IGNORE_LABEL_ID)
            teacher_counts = teacher_mask.sum(-1)
            valid_teacher = teacher_counts > 0
            if valid_teacher.any():
                teacher_divisor = teacher_counts.clamp_min(1).unsqueeze(-1)
                distill_loss = (
                    self.loss_fn(
                        logits_f,
                        teacher_tokens,
                        ignore_index=IGNORE_LABEL_ID,
                        valid_mask=teacher_mask,
                    )
                    / teacher_divisor
                ).sum()
                total_loss = total_loss + self.teacher_weight * distill_loss
                metrics["distill_loss"] = distill_loss.detach()

        if self.teacher_step_weight > 0:
            teacher_steps = new_carry.current_data.get("teacher_steps")
            if teacher_steps is not None:
                step_mask = teacher_steps >= 0
                if step_mask.any():
                    teacher_steps_f = teacher_steps.to(torch.float32)
                    student_steps = new_carry.steps.to(torch.float32)
                    step_diff = torch.abs(student_steps - teacher_steps_f)
                    step_loss = torch.where(step_mask, step_diff, torch.zeros_like(step_diff)).sum()
                    total_loss = total_loss + self.teacher_step_weight * step_loss
                    metrics["distill_step_loss"] = step_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
