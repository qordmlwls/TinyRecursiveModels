import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import hydra
import pydantic
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.recursive_reasoning.planner import TRMPlanner, TRMPlannerConfig
from pretrain import ArchConfig, LossConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy


class PlannerSettings(pydantic.BaseModel):
    beam_size: int = 4
    max_steps: int = 16
    max_nodes_per_beam: int = 64
    score_alpha: float = 1.0
    score_beta: float = 0.0
    step_penalty: float = 0.01
    num_runs: int = 1
    top_k_per_puzzle: int = 1
    noise_std: float = 0.0


class PlannerJobConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    checkpoint_path: str
    output_dir: str
    global_batch_size: int = 128
    seed: int = 0
    limit_batches: Optional[int] = None
    planner: PlannerSettings = PlannerSettings()


def _build_trm_model(cfg: PlannerJobConfig, metadata) -> torch.nn.Module:
    model_cfg = dict(
        **cfg.arch.__pydantic_extra__,  # type: ignore[arg-type]
        batch_size=cfg.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(cfg.arch.name)
    loss_cls = load_model_class(cfg.arch.loss.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_cls(model_cfg).to(device)
    wrapped = loss_cls(model, **cfg.arch.loss.__pydantic_extra__)  # type: ignore[arg-type]

    state_dict = torch.load(cfg.checkpoint_path, map_location=device)

    # Align puzzle embeddings if necessary
    puzzle_key = "_orig_mod.model.inner.puzzle_emb.weights"
    if puzzle_key in state_dict:
        expected = wrapped.model.inner.puzzle_emb.weights.shape  # type: ignore[attr-defined]
        if state_dict[puzzle_key].shape != expected:
            state_dict[puzzle_key] = (
                state_dict[puzzle_key].mean(dim=0, keepdim=True).expand(expected).contiguous()
            )

    wrapped.load_state_dict(state_dict, strict=False)
    wrapped.eval()
    return wrapped.model  # type: ignore[return-value]


def _extract_puzzle_id(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return int(tensor.item())
    return int(tensor.view(-1)[0].item())


def _save_entry(path: str, entry: Dict, top_k: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        prev = torch.load(path, weights_only=False)
    else:
        prev = {"puzzle_id": entry["puzzle_id"], "entries": []}

    entries = prev.get("entries", [])
    entries.append(entry)
    entries.sort(key=lambda e: e["score"], reverse=True)
    entries = entries[:top_k]
    prev["entries"] = entries
    prev["puzzle_id"] = entry["puzzle_id"]
    torch.save(prev, path)


@hydra.main(config_path="../config", config_name="cfg_trm_planner", version_base=None)
def main(hydra_cfg: DictConfig):
    job_cfg = PlannerJobConfig(**hydra_cfg)  # type: ignore[arg-type]
    os.makedirs(job_cfg.output_dir, exist_ok=True)

    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=job_cfg.seed,
            dataset_paths=job_cfg.data_paths,
            global_batch_size=job_cfg.global_batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split="train",
    )
    dataloader = DataLoader(dataset, batch_size=None)

    model = _build_trm_model(job_cfg, dataset.metadata)
    planner_cfg = TRMPlannerConfig(
        **job_cfg.planner.model_dump(exclude={"num_runs", "top_k_per_puzzle"})
    )
    planner = TRMPlanner(model, planner_cfg)

    summary = {"samples": 0}
    progress = tqdm(enumerate(dataloader), total=job_cfg.limit_batches)
    for batch_idx, (_, batch, _) in progress:
        batch = {k: v.to(planner.device) for k, v in batch.items()}

        for run_idx in range(job_cfg.planner.num_runs):
            nodes = planner.plan_batch(batch)

            for sample_idx, node in enumerate(nodes):
                logits = node.outputs["logits"]
                tokens = logits.argmax(dim=-1).to(torch.int32).cpu()
                labels = batch["labels"][sample_idx : sample_idx + 1]
                mask = (labels != IGNORE_LABEL_ID)
                ce = stablemax_cross_entropy(
                    logits,
                    labels,
                    ignore_index=IGNORE_LABEL_ID,
                    valid_mask=mask,
                )
                teacher_loss = (ce.sum() / mask.sum().clamp_min(1)).item()
                puzzle_id = _extract_puzzle_id(batch["puzzle_identifiers"][sample_idx].cpu())
                entry = {
                    "puzzle_id": puzzle_id,
                    "step": node.step,
                    "score": node.score,
                    "tokens": tokens.squeeze(0).numpy(),
                    "teacher_loss": teacher_loss,
                    "logits": logits.squeeze(0).to(torch.float16).cpu().numpy(),
                }
                out_path = os.path.join(job_cfg.output_dir, f"{puzzle_id}.pt")
                _save_entry(out_path, entry, job_cfg.planner.top_k_per_puzzle)
                summary["samples"] += 1

        progress.set_postfix(saved=summary["samples"])
        if job_cfg.limit_batches is not None and batch_idx + 1 >= job_cfg.limit_batches:
            break

    summary_path = os.path.join(job_cfg.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
