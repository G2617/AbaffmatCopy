# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import copy
import datetime
import hashlib
import logging
import os
import re
import sys
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from ml_collections.config_dict import ConfigDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from configs.config import parse_configs, parse_sys_args, save_config
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from datasets.dataloader import get_preference_dataloader
from losses.dpo import build_preference_loss, compute_preference_loss_metrics
from models.scoring_model import ComplexScoringModel
from runners.score_complex import download_inference_cache
from utils.distributed import DIST_WRAPPER
from utils.model_name import parse_protenix_model_name
from utils.seed import seed_everything
from utils.torch_utils import to_device


logger = logging.getLogger(__name__)
torch.serialization.add_safe_globals([Namespace])


OPTIONAL_EMPTY_DPO_ARGS = {
    "--load_checkpoint_path",
    "--reference_checkpoint_path",
}
EMPTY_DPO_ARG_SENTINEL = "__ABAFFMAT_EMPTY__"


def normalize_optional_empty_args(args: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(args):
        arg = args[index]
        normalized.append(arg)
        if (
            arg in OPTIONAL_EMPTY_DPO_ARGS
            and (index + 1 >= len(args) or args[index + 1].startswith("--"))
        ):
            normalized.append(EMPTY_DPO_ARG_SENTINEL)
            index += 1
            continue
        if arg in OPTIONAL_EMPTY_DPO_ARGS and index + 1 < len(args) and args[index + 1] == "":
            normalized.append(EMPTY_DPO_ARG_SENTINEL)
            index += 2
            continue
        index += 1
    return normalized


def clear_empty_dpo_arg_sentinels(configs: Any) -> None:
    for key in ("load_checkpoint_path", "reference_checkpoint_path"):
        if getattr(configs, key, None) in {
            EMPTY_DPO_ARG_SENTINEL,
            "None",
            "none",
            "null",
        }:
            setattr(configs, key, "")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def load_checkpoint_state(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


CORE_LOG_METRICS = [
    "loss",
    "pair_accuracy",
    "relative_margin",
    "policy_margin",
]


WANDB_DPO_METRICS = [
    "loss",
    "pair_accuracy",
    "policy_pair_accuracy",
    "reference_pair_accuracy",
    "relative_margin_accuracy",
    "policy_margin",
    "reference_margin",
    "relative_margin",
    "dpo_logit",
    "policy_chosen_mean",
    "policy_rejected_mean",
    "reference_chosen_mean",
    "reference_rejected_mean",
]


def update_metric_sums(
    metric_sums: dict[str, float],
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    weight: int,
) -> None:
    safe_weight = max(int(weight), 1)
    metric_sums["loss"] = metric_sums.get("loss", 0.0) + float(loss.item()) * safe_weight
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item()) * safe_weight


def average_metric_sums(
    metric_sums: dict[str, float],
    total_weight: int,
) -> dict[str, float]:
    if total_weight <= 0:
        return {}
    return {key: value / total_weight for key, value in metric_sums.items()}


def format_metric_fields(metrics: dict[str, float], keys: list[str]) -> str:
    return " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


def preference_group_label(pair_metadata: dict[str, Any]) -> str:
    return "|".join(
        str(pair_metadata.get(key, "")).strip()
        for key in ("library_name", "experimental_round", "antigen_name")
    )


def sanitize_metric_component(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")
    return normalized or "unknown"


def update_example_metric_sums(
    metric_sums: dict[str, float],
    example_metrics: dict[str, torch.Tensor],
) -> None:
    for key, values in example_metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(values.sum().item())


def update_group_metric_sums(
    group_metric_sums: dict[str, dict[str, float]],
    group_pair_counts: dict[str, int],
    pair_metadata: list[dict[str, Any]],
    example_metrics: dict[str, torch.Tensor],
) -> None:
    if not pair_metadata:
        return
    detached_metrics = {
        key: values.detach().reshape(-1).cpu()
        for key, values in example_metrics.items()
    }
    for pair_index, metadata in enumerate(pair_metadata):
        group_label = preference_group_label(metadata)
        group_pair_counts[group_label] = group_pair_counts.get(group_label, 0) + 1
        group_sums = group_metric_sums.setdefault(group_label, {})
        for key, values in detached_metrics.items():
            group_sums[key] = group_sums.get(key, 0.0) + float(values[pair_index].item())


def build_stratified_eval_subset_indices(
    pair_metadata: list[dict[str, Any]],
    total_pair_budget: int,
) -> list[int] | None:
    if total_pair_budget <= 0 or total_pair_budget >= len(pair_metadata):
        return None

    group_to_indices: dict[str, list[int]] = {}
    ordered_group_labels: list[str] = []
    for pair_index, metadata in enumerate(pair_metadata):
        group_label = preference_group_label(metadata)
        if group_label not in group_to_indices:
            group_to_indices[group_label] = []
            ordered_group_labels.append(group_label)
        group_to_indices[group_label].append(pair_index)

    if not ordered_group_labels:
        return []

    group_caps = {group_label: 0 for group_label in ordered_group_labels}
    base_cap = total_pair_budget // len(ordered_group_labels)
    remainder = total_pair_budget % len(ordered_group_labels)
    for index, group_label in enumerate(ordered_group_labels):
        requested_cap = base_cap + (1 if index < remainder else 0)
        group_caps[group_label] = min(requested_cap, len(group_to_indices[group_label]))

    assigned_pairs = sum(group_caps.values())
    remaining_budget = total_pair_budget - assigned_pairs
    while remaining_budget > 0:
        made_progress = False
        for group_label in ordered_group_labels:
            if remaining_budget <= 0:
                break
            if group_caps[group_label] >= len(group_to_indices[group_label]):
                continue
            group_caps[group_label] += 1
            remaining_budget -= 1
            made_progress = True
        if not made_progress:
            break

    selected_indices = []
    for group_label in ordered_group_labels:
        selected_indices.extend(group_to_indices[group_label][: group_caps[group_label]])
    return selected_indices


def average_group_metrics(
    group_metrics: dict[str, dict[str, float]],
    keys: list[str],
) -> dict[str, float]:
    if not group_metrics:
        return {}
    macro_metrics = {}
    for key in keys:
        values = [metrics[key] for metrics in group_metrics.values() if key in metrics]
        if values:
            macro_metrics[key] = sum(values) / len(values)
    return macro_metrics


def get_accumulation_steps(configs: Any) -> int:
    return max(int(getattr(configs, "iters_to_accumulate", 1)), 1)


def should_step_optimizer(micro_step: int, accumulation_steps: int) -> bool:
    return micro_step % accumulation_steps == 0


def build_optimizer(params: list[torch.nn.Parameter], configs: Any) -> torch.optim.Optimizer:
    adam_configs = getattr(configs, "adam", None)
    beta1 = float(getattr(adam_configs, "beta1", 0.9)) if adam_configs else 0.9
    beta2 = float(getattr(adam_configs, "beta2", 0.999)) if adam_configs else 0.999
    weight_decay = (
        float(getattr(adam_configs, "weight_decay", 0.0)) if adam_configs else 0.0
    )
    use_adamw = bool(getattr(adam_configs, "use_adamw", False)) if adam_configs else False
    optimizer_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    return optimizer_cls(
        params,
        lr=float(configs.lr),
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )


class DPOTrainer:
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_tracking()
        self.print("Initializing policy and reference models")
        self.init_models()
        self.print("Initializing DPO loss")
        self.init_loss()
        self.print("Initializing optimizer")
        self.init_optimizer()
        self.print("Building preference dataloaders")
        self.init_data()
        self.best_val_accuracy = float("-inf")
        self.bad_eval_count = 0

    def print(self, message: str) -> None:
        if DIST_WRAPPER.rank == 0:
            logger.info(message)

    def init_tracking(self) -> None:
        self.wandb_run = None
        if not self.configs.use_wandb or DIST_WRAPPER.rank != 0:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases tracking was requested with --use_wandb True, "
                "but wandb is not installed. Install it with `pip install wandb`."
            ) from exc

        config_payload = (
            self.configs.to_dict() if hasattr(self.configs, "to_dict") else dict(self.configs)
        )
        self.wandb_run = wandb.init(
            project=self.configs.project,
            name=self.run_name,
            id=(self.configs.wandb_id or None),
            resume=("allow" if self.configs.wandb_id else None),
            config=config_payload,
            dir=self.run_dir,
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.wandb_run is None or DIST_WRAPPER.rank != 0:
            return
        self.wandb_run.log(metrics, step=step)

    def init_env(self) -> None:
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            else:
                self.device = torch.device(f"cuda:{self.configs.gpuid}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            dist.init_process_group(
                backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
            )
        if not self.configs.deterministic_seed:
            hash_string = f"({self.configs.seed},{DIST_WRAPPER.rank},train_dpo)"
            rank_seed = int(hashlib.sha256(hash_string.encode("utf8")).hexdigest(), 16)
            rank_seed = rank_seed % (2**32)
        else:
            rank_seed = self.configs.seed
        seed_everything(seed=rank_seed, deterministic=self.configs.deterministic)

    def init_basics(self) -> None:
        self.step = 0
        self.global_step = 0
        self.run_name = self.configs.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.log_path = f"{self.run_dir}/train_dpo.log"
        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            save_config(self.configs, os.path.join(self.run_dir, "config.yaml"))
            self.log = open(self.log_path, "w", buffering=1)

    def write_log(self, text: str) -> None:
        if DIST_WRAPPER.rank == 0:
            print(text)
            self.log.write(text + "\n")

    def load_backbone_checkpoint(self, model: ComplexScoringModel) -> None:
        checkpoint_path = f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        checkpoint = load_checkpoint_state(checkpoint_path, self.device)
        state_dict = checkpoint["model"]
        sample_key = next(iter(state_dict))
        if sample_key.startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        model.protenix_model.load_state_dict(state_dict=state_dict, strict=self.configs.load_strict)

    def load_full_model_checkpoint(
        self,
        model: ComplexScoringModel,
        checkpoint_path: str,
        load_optimizer: bool = False,
    ) -> dict[str, Any]:
        checkpoint = load_checkpoint_state(checkpoint_path, self.device)
        state_dict = checkpoint["model"]
        sample_key = next(iter(state_dict))
        if sample_key.startswith("module."):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict=state_dict, strict=self.configs.load_strict)
        return checkpoint if load_optimizer else {}

    def init_models(self) -> None:
        self.policy_model = ComplexScoringModel(self.configs).to(self.device)
        self.reference_model = ComplexScoringModel(self.configs).to(self.device)
        self.load_backbone_checkpoint(self.policy_model)
        self.load_backbone_checkpoint(self.reference_model)

        optimizer_checkpoint = None
        if self.configs.load_checkpoint_path:
            optimizer_checkpoint = self.load_full_model_checkpoint(
                self.policy_model,
                self.configs.load_checkpoint_path,
                load_optimizer=(not self.configs.load_params_only),
            )
        if self.configs.reference_checkpoint_path:
            self.load_full_model_checkpoint(
                self.reference_model,
                self.configs.reference_checkpoint_path,
                load_optimizer=False,
            )
        else:
            self.reference_model.load_state_dict(
                copy.deepcopy(self.policy_model.state_dict()),
                strict=self.configs.load_strict,
            )

        if self.configs.freeze_backbone or self.configs.train_head_only:
            for parameter in self.policy_model.protenix_model.parameters():
                parameter.requires_grad = False

        for parameter in self.reference_model.parameters():
            parameter.requires_grad = False
        self.reference_model.eval()

        self.use_ddp = False
        if DIST_WRAPPER.world_size > 1:
            self.use_ddp = True
            self.policy_model = DDP(
                self.policy_model,
                find_unused_parameters=self.configs.find_unused_parameters,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=False,
            )
        self.optimizer_checkpoint = optimizer_checkpoint

    def init_loss(self) -> None:
        self.dpo_loss_type = str(getattr(self.configs, "dpo_loss_type", "anchored_dpo"))
        self.uses_reference_scores = self.dpo_loss_type.strip().lower() in {
            "anchored_dpo",
            "dpo",
        }
        self.loss = build_preference_loss(
            loss_type=self.dpo_loss_type,
            beta=self.configs.dpo_beta,
        )
        self.scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.dtype == "fp16"),
        )

    def init_optimizer(self) -> None:
        self.trainable_params = [
            param for param in self.policy_model.parameters() if param.requires_grad
        ]
        self.accumulation_steps = get_accumulation_steps(self.configs)
        self.micro_step = 0
        self.optimizer = build_optimizer(self.trainable_params, self.configs)
        if self.optimizer_checkpoint and not self.configs.skip_load_optimizer:
            try:
                self.optimizer.load_state_dict(self.optimizer_checkpoint["optimizer"])
                self.step = int(self.optimizer_checkpoint["step"]) + 1
                self.global_step = int(
                    self.optimizer_checkpoint.get(
                        "global_step",
                        self.step * self.accumulation_steps,
                    )
                )
                self.micro_step = int(
                    self.optimizer_checkpoint.get("micro_step", self.global_step)
                )
            except (KeyError, ValueError) as exc:
                self.print(f"Skipping optimizer state load due to checkpoint mismatch: {exc}")

    def init_data(self) -> None:
        self.train_dl = get_preference_dataloader(
            self.configs,
            canonical_path=self.configs.canonical_output_path,
            json_output_dir=self.configs.json_output_dir,
            split="train",
            batchsize=self.configs.batchsize,
            shuffle=True,
            split_path=self.configs.train_split_path,
        )
        self.val_dl = get_preference_dataloader(
            self.configs,
            canonical_path=self.configs.canonical_output_path,
            json_output_dir=self.configs.json_output_dir,
            split="val",
            batchsize=self.configs.batchsize,
            shuffle=False,
            split_path=self.configs.val_split_path,
        )
        train_pairs = len(self.train_dl.dataset)
        val_pairs = len(self.val_dl.dataset)
        train_samples = len(self.train_dl.dataset.canonical_df)
        val_samples = len(self.val_dl.dataset.canonical_df)
        data_source = "json_output_dir" if self.configs.json_output_dir else "canonical_table"
        self.print(
            "Preference data ready: "
            + f"source={data_source} train_samples={train_samples} train_pairs={train_pairs} "
            + f"val_samples={val_samples} val_pairs={val_pairs} "
            + f"batchsize={self.configs.batchsize} num_workers={self.configs.num_workers}"
        )
        if train_pairs == 0:
            self.print(
                "Training split produced zero preference pairs. "
                "Check preference_gap, split manifests, or canonicalized sample counts."
            )
        if val_pairs == 0:
            self.print(
                "Validation split produced zero preference pairs. "
                "Evaluation will report zeros until more validation pairs are available."
            )

    def precision_context(self):
        precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        if not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=precision, cache_enabled=False)

    def forward_scores(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.policy_model(input_feature_dict=batch["input_feature_dict"]).reshape(-1)

    @torch.no_grad()
    def forward_reference_scores(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.reference_model(input_feature_dict=batch["input_feature_dict"]).reshape(-1)

    def compute_loss(
        self,
        batch: dict[str, Any],
        return_example_metrics: bool = False,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor] | None,
    ]:
        chosen_batch = to_device(batch["chosen_batch"], self.device)
        rejected_batch = to_device(batch["rejected_batch"], self.device)
        with self.precision_context():
            policy_chosen = self.forward_scores(chosen_batch)
            policy_rejected = self.forward_scores(rejected_batch)
            if self.uses_reference_scores:
                reference_chosen = self.forward_reference_scores(chosen_batch)
                reference_rejected = self.forward_reference_scores(rejected_batch)
            else:
                reference_chosen = None
                reference_rejected = None
            loss, metrics, example_metrics = compute_preference_loss_metrics(
                loss_type=self.dpo_loss_type,
                beta=self.configs.dpo_beta,
                policy_chosen=policy_chosen,
                policy_rejected=policy_rejected,
                reference_chosen=reference_chosen,
                reference_rejected=reference_rejected,
                return_example_metrics=return_example_metrics,
            )
        return loss, metrics, example_metrics

    def save_checkpoint(self, name: str) -> None:
        if DIST_WRAPPER.rank != 0:
            return
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": unwrap_model(self.policy_model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "global_step": self.global_step,
                "micro_step": self.micro_step,
            },
            checkpoint_path,
        )
        self.print(f"Saved checkpoint to {checkpoint_path}")
        if self.wandb_run is not None and DIST_WRAPPER.rank == 0:
            self.wandb_run.summary[f"checkpoint/{name}"] = checkpoint_path

    @torch.no_grad()
    def evaluate(
        self,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, int],
    ]:
        unwrap_model(self.policy_model).eval()
        total_batches = 0
        total_pairs = 0
        metric_sums: dict[str, float] = {}
        group_metric_sums: dict[str, dict[str, float]] = {}
        group_pair_counts: dict[str, int] = {}
        max_eval_batches = int(getattr(self.configs, "max_eval_batches", -1))
        eval_log_interval = int(getattr(self.configs, "eval_log_interval", 10))
        eval_pair_budget = (
            max_eval_batches * int(self.configs.batchsize) if max_eval_batches > 0 else -1
        )
        subset_indices = build_stratified_eval_subset_indices(
            self.val_dl.dataset.pairs,
            eval_pair_budget,
        )
        if subset_indices is None:
            eval_loader = self.val_dl
            total_target_pairs = len(self.val_dl.dataset)
            group_count = len(
                {
                    preference_group_label(pair)
                    for pair in self.val_dl.dataset.pairs
                }
            )
        else:
            subset_dataset = Subset(self.val_dl.dataset, subset_indices)
            dataloader_kwargs = {
                "dataset": subset_dataset,
                "batch_size": int(self.configs.batchsize),
                "shuffle": False,
                "collate_fn": self.val_dl.collate_fn,
                "num_workers": int(self.configs.num_workers),
                "drop_last": False,
            }
            if int(self.configs.num_workers) > 0:
                dataloader_kwargs["prefetch_factor"] = 2
            eval_loader = DataLoader(**dataloader_kwargs)
            total_target_pairs = len(subset_indices)
            group_count = len(
                {
                    preference_group_label(self.val_dl.dataset.pairs[index])
                    for index in subset_indices
                }
            )
        self.print(
            "Starting validation: "
            + f"groups={group_count} "
            + f"target_pairs={total_target_pairs} "
            + f"max_eval_batches={max_eval_batches}"
        )
        for batch in eval_loader:
            loss, metrics, example_metrics = self.compute_loss(
                batch,
                return_example_metrics=True,
            )
            if example_metrics is None:
                raise RuntimeError("validation expected per-example metrics")
            update_example_metric_sums(metric_sums, example_metrics)
            update_group_metric_sums(
                group_metric_sums,
                group_pair_counts,
                batch["pair_metadata"],
                example_metrics,
            )
            total_pairs += len(batch["pair_metadata"])
            total_batches += 1
            if eval_log_interval > 0 and total_batches % eval_log_interval == 0:
                progress_metrics = average_metric_sums(metric_sums, total_pairs)
                self.print(
                    "Validation progress: "
                    + f"batches={total_batches} selected_pairs={total_pairs}/{total_target_pairs} "
                    + format_metric_fields(progress_metrics, CORE_LOG_METRICS)
                )
        unwrap_model(self.policy_model).train()
        if total_pairs == 0:
            return (
                {key: 0.0 for key in WANDB_DPO_METRICS},
                {key: 0.0 for key in WANDB_DPO_METRICS},
                {},
                {},
            )
        group_metrics = {
            group_label: average_metric_sums(group_sums, group_pair_counts[group_label])
            for group_label, group_sums in group_metric_sums.items()
        }
        macro_metrics = average_group_metrics(group_metrics, WANDB_DPO_METRICS)
        return (
            average_metric_sums(metric_sums, total_pairs),
            macro_metrics,
            group_metrics,
            group_pair_counts,
        )

    def maybe_evaluate(self, force: bool = False) -> None:
        if (
            not force
            and (
                self.configs.eval_interval <= 0
                or self.step % self.configs.eval_interval != 0
            )
        ):
            return
        metrics, macro_metrics, group_metrics, group_pair_counts = self.evaluate()
        self.write_log(
            "VAL_MICRO "
            + f"step={self.step} "
            + format_metric_fields(
                metrics,
                [
                    "loss",
                    "pair_accuracy",
                    "relative_margin_accuracy",
                    "relative_margin",
                    "policy_margin",
                    "reference_margin",
                ],
            )
        )
        self.write_log(
            "VAL_MACRO "
            + f"step={self.step} "
            + format_metric_fields(
                macro_metrics,
                [
                    "loss",
                    "pair_accuracy",
                    "relative_margin_accuracy",
                    "relative_margin",
                    "policy_margin",
                    "reference_margin",
                ],
            )
        )
        self.log_metrics(
            {f"val/{key}": metrics[key] for key in WANDB_DPO_METRICS if key in metrics},
            self.step,
        )
        self.log_metrics(
            {
                f"val_macro/{key}": macro_metrics[key]
                for key in WANDB_DPO_METRICS
                if key in macro_metrics
            },
            self.step,
        )
        for group_label in sorted(group_metrics):
            group_summary = group_metrics[group_label]
            self.write_log(
                "VAL_GROUP "
                + f"step={self.step} "
                + f"group={group_label} "
                + f"pairs={group_pair_counts.get(group_label, 0)} "
                + format_metric_fields(
                    group_summary,
                    [
                        "loss",
                        "pair_accuracy",
                        "relative_margin_accuracy",
                        "relative_margin",
                        "policy_margin",
                        "reference_margin",
                    ],
                )
            )
            safe_group_label = sanitize_metric_component(group_label)
            self.log_metrics(
                {
                    f"val_by_group/{safe_group_label}/{key}": group_summary[key]
                    for key in WANDB_DPO_METRICS
                    if key in group_summary
                }
                | {f"val_by_group/{safe_group_label}/pairs": float(group_pair_counts[group_label])},
                self.step,
            )
        if metrics["pair_accuracy"] > self.best_val_accuracy:
            self.best_val_accuracy = metrics["pair_accuracy"]
            self.bad_eval_count = 0
            self.save_checkpoint("best.pt")
        else:
            self.bad_eval_count += 1
        if self.bad_eval_count >= self.configs.patience:
            raise StopIteration("early stopping triggered")

    def run(self) -> None:
        unwrap_model(self.policy_model).train()
        if len(self.train_dl) == 0:
            raise RuntimeError("training dataset produced zero preference pairs")
        self.print(
            "Starting DPO training loop. "
            + f"dpo_loss_type={self.dpo_loss_type} "
            + f"log_interval={self.configs.log_interval} "
            + f"eval_interval={self.configs.eval_interval} "
            + f"max_steps={self.configs.max_steps} "
            + f"iters_to_accumulate={self.accumulation_steps} "
            + f"grad_clip_norm={self.configs.grad_clip_norm}"
        )
        if self.configs.eval_first and self.step == 0:
            self.print("Running initial validation before training")
            self.maybe_evaluate(force=True)
        self.print(
            "Fetching the first preference batch. If you are running in Colab, this can take a while "
            "because features are built lazily on the CPU."
        )
        epoch = 0
        train_metric_sums: dict[str, float] = {}
        train_metric_pairs = 0
        while self.step < self.configs.max_steps:
            if hasattr(self.train_dl.sampler, "set_epoch"):
                self.train_dl.sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(self.train_dl):
                if self.step == 0 and batch_index == 0:
                    self.print(
                        "First preference batch received: "
                        + f"pairs={len(batch['pair_metadata'])}"
                    )
                if self.micro_step % self.accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                loss, metrics, _ = self.compute_loss(batch)
                batch_pairs = len(batch["pair_metadata"])
                update_metric_sums(train_metric_sums, loss, metrics, batch_pairs)
                train_metric_pairs += batch_pairs
                scaled_loss = loss / self.accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                self.micro_step += 1
                self.global_step += 1
                if not should_step_optimizer(self.micro_step, self.accumulation_steps):
                    continue
                grad_clip_norm = float(getattr(self.configs, "grad_clip_norm", 0.0))
                if grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_params,
                        max_norm=grad_clip_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                if self.step == 1:
                    self.print("First optimization step completed")
                if (
                    self.configs.log_interval > 0
                    and self.step % self.configs.log_interval == 0
                ):
                    train_metrics = average_metric_sums(train_metric_sums, train_metric_pairs)
                    self.write_log(
                        "TRAIN "
                        + f"step={self.step} "
                        + format_metric_fields(
                            train_metrics,
                            [
                                "loss",
                                "pair_accuracy",
                                "relative_margin_accuracy",
                                "relative_margin",
                                "policy_margin",
                                "reference_margin",
                            ],
                        )
                    )
                    train_log_payload = {
                        f"train/{key}": train_metrics[key]
                        for key in WANDB_DPO_METRICS
                        if key in train_metrics
                    }
                    train_log_payload["train/lr"] = float(
                        self.optimizer.param_groups[0]["lr"]
                    )
                    self.log_metrics(train_log_payload, self.step)
                    train_metric_sums = {}
                    train_metric_pairs = 0
                if (
                    self.configs.checkpoint_interval > 0
                    and self.step % self.configs.checkpoint_interval == 0
                ):
                    self.save_checkpoint(f"step_{self.step}.pt")
                self.maybe_evaluate()
                if self.step >= self.configs.max_steps:
                    break
            epoch += 1
        self.save_checkpoint("last.pt")
        if self.wandb_run is not None and DIST_WRAPPER.rank == 0:
            self.wandb_run.finish()


def main() -> None:
    log_format = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    configs_base["triangle_attention"] = os.environ.get("TRIANGLE_ATTENTION", "triattention")
    configs_base["triangle_multiplicative"] = os.environ.get("TRIANGLE_MULTIPLICATIVE", "cuequivariance")
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    original_args = sys.argv[1:]
    sys.argv = [sys.argv[0], *normalize_optional_empty_args(original_args)]
    arg_str = parse_sys_args()
    configs = parse_configs(configs=configs, arg_str=arg_str, fill_required_with_null=True)
    clear_empty_dpo_arg_sentinels(configs)
    if "--model_name" not in arg_str:
        configs.model_name = "protenix_mini_default_v0.5.0"
    if "--load_checkpoint_path" not in original_args:
        configs.load_checkpoint_path = ""
    model_name = configs.model_name
    configs.use_msa = False
    download_inference_cache(configs)
    parsed_model_name = parse_protenix_model_name(model_name)
    if parsed_model_name is None:
        logger.info("Initializing DPO training with model_name=%s", model_name)
    else:
        logger.info(
            "Initializing DPO training with model_name=%s (size=%s, feature=%s, version=%s)",
            model_name,
            parsed_model_name["model_size"],
            parsed_model_name["model_feature"],
            parsed_model_name["model_version"],
        )
    model_specific_configs = ConfigDict(model_configs[model_name])
    configs.update(model_specific_configs)
    trainer = DPOTrainer(configs)
    trainer.run()


if __name__ == "__main__":
    main()
