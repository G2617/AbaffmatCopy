# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import argparse
import copy
import importlib.util
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from configs.extend_types import get_bool_value
from datasets.preference_utils import build_preference_pairs
from datasets.yeast_display import load_dataframe
from utils.sample_ids import load_ordered_sample_ids


SCORE_INPUT_DIRNAME = "canonical_holdout_json"


def parse_bool_arg(raw_value: str | bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return get_bool_value(str(raw_value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one or more checkpoints on the exact same held-out yeast-display "
            "samples and write comparable score tables plus summary metrics."
        )
    )
    parser.add_argument("--canonical_output_path", required=True)
    parser.add_argument(
        "--json_output_dir",
        default="",
        help=(
            "Optional directory of pre-generated per-sample JSON files. If omitted, "
            "benchmark inputs are generated from canonical_output_path and holdout_ids_path."
        ),
    )
    parser.add_argument("--holdout_ids_path", required=True)
    parser.add_argument("--checkpoint_paths", required=True, nargs="+")
    parser.add_argument("--output_dir", default="outputs/yeast_display_benchmark")
    parser.add_argument("--project", default="yeast-display-benchmark")
    parser.add_argument("--run_prefix", default="yeast-display-benchmark")
    parser.add_argument("--use_wandb", nargs="?", const="True", default=False, type=parse_bool_arg)
    parser.add_argument("--wandb_id", default="")
    parser.add_argument("--model_name", default="protenix_mini_default_v0.5.0")
    parser.add_argument("--load_checkpoint_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--precompute_esm", action="store_true")
    parser.add_argument("--preference_gap", type=float, default=0.5)
    parser.add_argument("--top_ks", default="1,5,10,20")
    parser.add_argument("--winner_count", type=int, default=10)
    parser.add_argument("--ranking_label_column", default="label_raw")
    return parser.parse_args()


def parse_top_ks(raw_value: str) -> list[int]:
    top_ks = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("top_ks must contain only positive integers")
        if value not in top_ks:
            top_ks.append(value)
    if not top_ks:
        raise ValueError("top_ks must contain at least one positive integer")
    return top_ks


def load_holdout_dataframe(
    canonical_output_path: str,
    holdout_ids_path: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the canonical yeast-display dataframe and filter to the held-out samples in the specified order.
    """
    canonical_df = load_dataframe(canonical_output_path)
    ordered_ids = load_ordered_sample_ids(holdout_ids_path)
    if not ordered_ids:
        raise ValueError("holdout_ids_path did not contain any sample IDs")
    sample_id_set = set(ordered_ids)
    available_ids = set(canonical_df["sample_id"].tolist())
    missing_ids = [sample_id for sample_id in ordered_ids if sample_id not in available_ids]
    if missing_ids:
        raise ValueError(f"Holdout IDs missing from canonical table: {missing_ids[:10]}")
    holdout_df = canonical_df[canonical_df["sample_id"].isin(sample_id_set)].copy()
    holdout_df["holdout_order"] = holdout_df["sample_id"].map({sample_id: idx for idx, sample_id in enumerate(ordered_ids)})
    holdout_df = holdout_df.sort_values("holdout_order").drop(columns=["holdout_order"]).reset_index(drop=True)
    return holdout_df, ordered_ids


def row_to_score_input(row: Any) -> dict[str, Any]:
    antigen_sequence = str(row["antigen_sequence"]).strip()
    if not antigen_sequence:
        raise ValueError(f"Missing antigen_sequence for sample {row['sample_id']}")
    return {
        "name": row["sample_id"],
        "sequences": [
            {"proteinChain": {"sequence": antigen_sequence, "count": 1}},
            {"proteinChain": {"sequence": row["sequence_aa_heavy"], "count": 1}},
            {"proteinChain": {"sequence": row["sequence_aa_light"], "count": 1}},
        ],
    }


def verify_score_input_jsons(json_output_dir: str, ordered_ids: list[str]) -> str:
    if not json_output_dir:
        raise ValueError("json_output_dir must be non-empty when verifying JSON inputs")
    json_dir = Path(json_output_dir)
    missing_json = [
        sample_id
        for sample_id in ordered_ids
        if not (json_dir / f"{sample_id}.json").exists()
    ]
    if missing_json:
        raise FileNotFoundError(f"Holdout IDs missing JSON payloads: {missing_json[:10]}")
    return str(json_dir)


def write_holdout_score_inputs(
    holdout_df: pd.DataFrame,
    output_dir: str | Path,
) -> str:
    json_dir = Path(output_dir) / SCORE_INPUT_DIRNAME
    json_dir.mkdir(parents=True, exist_ok=True)
    for row in holdout_df.to_dict("records"):
        sample = row_to_score_input(row)
        with open(json_dir / f"{sample['name']}.json", "w") as handle:
            json.dump([sample], handle)
    return str(json_dir)


def resolve_score_input_json_dir(
    args: argparse.Namespace,
    holdout_df: pd.DataFrame,
    ordered_ids: list[str],
    output_dir: Path,
) -> str:
    if args.json_output_dir:
        return verify_score_input_jsons(args.json_output_dir, ordered_ids)
    return write_holdout_score_inputs(holdout_df, output_dir)


def normalize_score_dict(score_dict: dict[str, Any]) -> dict[str, float]:
    """
    Ensure that the raw score dict from the runner is a mapping from sample_id to scalar float.
    """
    normalized = {}
    for sample_id, raw_score in score_dict.items():
        score_array = np.asarray(raw_score, dtype=float).reshape(-1)
        if score_array.size != 1:
            raise ValueError(f"Expected a scalar prediction for {sample_id}, got shape {score_array.shape}")
        normalized[sample_id] = float(score_array[0])
    return normalized


def compute_pairwise_accuracy(
    scored_df: pd.DataFrame,
    preference_gap: float,
) -> tuple[float | None, int]:
    """
    Compute the pairwise accuracy of the model scores given the experimental labels and the specified preference gap.
    """
    pair_records = build_preference_pairs(
        scored_df,
        preference_gap=preference_gap,
        pair_sampling_strategy="all",
        max_pairs_per_group=0,
    )
    if not pair_records:
        return None, 0
    score_by_sample_id = dict(zip(scored_df["sample_id"], scored_df["model_score"]))
    correct_count = sum(
        score_by_sample_id[pair["chosen_id"]] > score_by_sample_id[pair["rejected_id"]]
        for pair in pair_records
    )
    return correct_count / len(pair_records), len(pair_records)


def compute_topk_metrics(
    scored_df: pd.DataFrame,
    ranking_label_column: str,
    top_ks: list[int],
) -> dict[str, float]:
    """
    Compute top-k overlap, recovery, precision, and enrichment metrics for the scored dataframe.
    Returns a dict mapping metric names to values, where metric names are of the form "top_{k}_overlap", "top_{k}_recovery", "top_{k}_precision", and "top_{k}_enrichment".
    """
    metrics = {}
    n_rows = len(scored_df)
    experimental_ranked = scored_df.sort_values(ranking_label_column, ascending=False, kind="mergesort")
    model_ranked = scored_df.sort_values("model_score", ascending=False, kind="mergesort")
    for top_k in top_ks:
        effective_k = min(top_k, n_rows)
        experimental_ids = set(experimental_ranked.head(effective_k)["sample_id"].tolist())
        model_ids = set(model_ranked.head(effective_k)["sample_id"].tolist())
        overlap = len(experimental_ids & model_ids)
        recovery = overlap / effective_k if effective_k else math.nan
        precision = overlap / effective_k if effective_k else math.nan
        baseline_rate = effective_k / n_rows if n_rows else math.nan
        enrichment = precision / baseline_rate if baseline_rate else math.nan
        metrics[f"top_{top_k}_overlap"] = float(overlap)
        metrics[f"top_{top_k}_recovery"] = recovery
        metrics[f"top_{top_k}_precision"] = precision
        metrics[f"top_{top_k}_enrichment"] = enrichment
    return metrics


def compute_winner_rank_metrics(
    scored_df: pd.DataFrame,
    ranking_label_column: str,
    winner_count: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute winner rank metrics for the scored dataframe, where winners are defined as the top `winner_count` samples by the experimental ranking.
    Returns a tuple of (metrics_dict, winner_rank_df) where metrics_dict contains scalar metrics about the winner ranks and winner_rank_df contains the sample_id, experimental_rank, model_rank, model_score, and experimental label for each winner.
    """
    effective_winner_count = min(max(winner_count, 1), len(scored_df))
    experimental_ranked = scored_df.sort_values(ranking_label_column, ascending=False, kind="mergesort").reset_index(drop=True)
    model_ranked = scored_df.sort_values("model_score", ascending=False, kind="mergesort").reset_index(drop=True)
    model_rank_by_id = {
        sample_id: rank
        for rank, sample_id in enumerate(model_ranked["sample_id"].tolist(), start=1)
    }
    winner_rows = []
    for experimental_rank, row in enumerate(
        experimental_ranked.head(effective_winner_count).to_dict("records"),
        start=1,
    ):
        winner_rows.append(
            {
                "sample_id": row["sample_id"],
                "experimental_rank": experimental_rank,
                "model_rank": model_rank_by_id[row["sample_id"]],
                "model_score": float(row["model_score"]),
                "label_raw": float(row["label_raw"]),
                "label_std": float(row["label_std"]),
            }
        )
    winner_rank_df = pd.DataFrame(winner_rows)
    winner_ranks = winner_rank_df["model_rank"].astype(float).to_numpy()
    metrics = {
        "winner_count_evaluated": float(effective_winner_count),
        "best_winner_rank": float(np.min(winner_ranks)),
        "mean_winner_rank": float(np.mean(winner_ranks)),
        "median_winner_rank": float(np.median(winner_ranks)),
    }
    return metrics, winner_rank_df


def compute_summary_metrics(
    scored_df: pd.DataFrame,
    ranking_label_column: str,
    preference_gap: float,
    top_ks: list[int],
    winner_count: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Compute summary metrics for a scored dataframe, including pairwise accuracy, Spearman correlations, top-k overlaps/enrichment, and winner ranks.
    Returns a tuple of (summary_dict, winner_rank_df) where summary_dict contains scalar metrics and winner_rank_df contains the ranks of the top experimental winners.
    """
    if ranking_label_column not in scored_df.columns:
        raise ValueError(f"ranking_label_column not found in scored frame: {ranking_label_column}")
    pairwise_accuracy, pair_count = compute_pairwise_accuracy(scored_df, preference_gap=preference_gap)
    spearman_label_raw = scored_df["model_score"].corr(scored_df["label_raw"], method="spearman")
    spearman_label_std = scored_df["model_score"].corr(scored_df["label_std"], method="spearman")
    summary = {
        "n_samples": int(len(scored_df)),
        "pair_count": int(pair_count),
        "pairwise_accuracy": pairwise_accuracy,
        "spearman_label_raw": None if pd.isna(spearman_label_raw) else float(spearman_label_raw),
        "spearman_label_std": None if pd.isna(spearman_label_std) else float(spearman_label_std),
    }
    summary.update(compute_topk_metrics(scored_df, ranking_label_column=ranking_label_column, top_ks=top_ks))
    winner_metrics, winner_rank_df = compute_winner_rank_metrics(
        scored_df,
        ranking_label_column=ranking_label_column,
        winner_count=winner_count,
    )
    summary.update(winner_metrics)
    return summary, winner_rank_df


def build_score_configs(args: argparse.Namespace, checkpoint_path: str) -> Any:
    from configs.config import ConfigManager
    from ml_collections.config_dict import ConfigDict

    local_configs_base = copy.deepcopy(configs_base)
    checkpoint_label_value = checkpoint_label(checkpoint_path)
    local_configs_base["triangle_attention"] = os.environ.get("TRIANGLE_ATTENTION", "triattention")
    local_configs_base["triangle_multiplicative"] = os.environ.get("TRIANGLE_MULTIPLICATIVE", "cuequivariance")
    if (
        local_configs_base["triangle_multiplicative"] == "cuequivariance"
        and importlib.util.find_spec("cuequivariance_torch") is None
    ):
        local_configs_base["triangle_multiplicative"] = "torch"
    local_inference_configs = copy.deepcopy(inference_configs)
    local_inference_configs["input_json_path"] = getattr(
        args,
        "score_input_json_path",
        args.json_output_dir,
    )
    base_config = {
        **local_configs_base,
        **{"data": copy.deepcopy(data_configs)},
        **local_inference_configs,
    }
    config_overrides = {
        "project": args.project,
        "run_name": f"{args.run_prefix}_{checkpoint_label_value}",
        "base_dir": args.output_dir,
        "eval_interval": "-1",
        "log_interval": "-1",
        "max_steps": "0",
        "input_json_path": local_inference_configs["input_json_path"],
        "model_name": args.model_name,
        "load_checkpoint_dir": args.load_checkpoint_dir,
        "load_checkpoint_path": checkpoint_path,
        "load_params_only": "True",
        "skip_load_optimizer": "True",
        "skip_load_step": "True",
        "seed": str(args.seed),
        "deterministic_seed": "True",
        "dtype": args.dtype,
        "batchsize": str(args.batchsize),
        "num_workers": str(args.num_workers),
        "gpuid": str(args.gpuid),
        "precompute_esm": "True" if args.precompute_esm else "False",
        "score_input_json_path": local_inference_configs["input_json_path"],
        "score_sample_ids_path": args.holdout_ids_path,
        "use_msa": "False",
    }
    configs = ConfigManager(base_config).merge_configs(config_overrides)
    configs.update(ConfigDict(model_configs[args.model_name]))
    return configs


def score_checkpoint(args: argparse.Namespace, checkpoint_path: str) -> tuple[dict[str, float], str]:
    """
    Run the scoring runner on the specified checkpoint and return the raw score dict plus the run directory.
    """
    from runners.score_complex import ComplexScoringRunner, download_inference_cache

    configs = build_score_configs(args, checkpoint_path)
    download_inference_cache(configs)
    # Initialize the runner
    runner = ComplexScoringRunner(configs)
    try:
        score_dict = normalize_score_dict(runner.score_samples()) # Score the held-out samples with the loaded checkpoint
        if hasattr(runner, "log") and runner.log:
            runner.log.flush()
        output_path = Path(runner.score_output_dir) / "scores.pkl"
        with open(output_path, "wb") as handle:
            pickle.dump(score_dict, handle)
        return score_dict, runner.run_dir
    finally:
        if hasattr(runner, "log") and runner.log:
            runner.log.close()


def write_checkpoint_outputs(
    run_dir: str,
    checkpoint_path: str,
    scored_df: pd.DataFrame,
    summary: dict[str, Any],
    winner_rank_df: pd.DataFrame,
) -> dict[str, Any]:
    benchmark_dir = Path(run_dir) / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(benchmark_dir / "heldout_scores.csv", index=False)
    winner_rank_df.to_csv(benchmark_dir / "winner_ranks.csv", index=False)
    summary_record = {"checkpoint_path": checkpoint_path, "run_dir": run_dir, **summary}
    with open(benchmark_dir / "summary.json", "w") as handle:
        json.dump(summary_record, handle, indent=2, sort_keys=True)
    return summary_record


def checkpoint_label(checkpoint_path: str) -> str:
    return Path(checkpoint_path).stem if checkpoint_path else "base"


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


def wandb_metric_payload(summary_record: dict[str, Any]) -> dict[str, float]:
    return {
        f"benchmark/{key}": float(value)
        for key, value in summary_record.items()
        if key not in {"checkpoint_path", "run_dir"} and is_finite_number(value)
    }


def init_wandb(args: argparse.Namespace) -> tuple[Any | None, Any | None]:
    if not args.use_wandb:
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Weights & Biases tracking was requested with --use_wandb, "
            "but wandb is not installed. Install it with `pip install wandb`."
        ) from exc

    config_payload = dict(vars(args))
    run = wandb.init(
        project=args.project,
        name=args.run_prefix,
        id=(args.wandb_id or None),
        resume=("allow" if args.wandb_id else None),
        config=config_payload,
        dir=args.output_dir,
    )
    return run, wandb


def log_checkpoint_to_wandb(
    wandb_run: Any | None,
    summary_record: dict[str, Any],
    checkpoint_index: int,
) -> None:
    if wandb_run is None:
        return
    payload = wandb_metric_payload(summary_record)
    payload["benchmark/checkpoint_index"] = float(checkpoint_index)
    wandb_run.log(payload, step=checkpoint_index)
    summary_prefix = f"checkpoint_{checkpoint_index}"
    wandb_run.summary[f"{summary_prefix}/label"] = checkpoint_label(
        summary_record["checkpoint_path"]
    )
    wandb_run.summary[f"{summary_prefix}/path"] = summary_record["checkpoint_path"]
    for key, value in summary_record.items():
        if is_finite_number(value):
            wandb_run.summary[f"{summary_prefix}/{key}"] = float(value)


def log_summary_table_to_wandb(
    wandb_run: Any | None,
    wandb_module: Any | None,
    summary_records: list[dict[str, Any]],
) -> None:
    if wandb_run is None or wandb_module is None or not summary_records:
        return
    summary_df = pd.DataFrame(summary_records)
    wandb_run.log({"benchmark/summary_table": wandb_module.Table(dataframe=summary_df)})


def main() -> None:
    args = parse_args()
    top_ks = parse_top_ks(args.top_ks)
    holdout_df, ordered_ids = load_holdout_dataframe(
        canonical_output_path=args.canonical_output_path,
        holdout_ids_path=args.holdout_ids_path,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.score_input_json_path = resolve_score_input_json_dir(
        args,
        holdout_df=holdout_df,
        ordered_ids=ordered_ids,
        output_dir=output_dir,
    )
    wandb_run, wandb_module = init_wandb(args)
    summary_records = []
    try:
        for checkpoint_index, checkpoint_path in enumerate(args.checkpoint_paths):
            score_dict, run_dir = score_checkpoint(args, checkpoint_path=checkpoint_path)
            missing_predictions = [
                sample_id for sample_id in ordered_ids if sample_id not in score_dict
            ]
            if missing_predictions:
                raise ValueError(
                    f"Missing predictions for held-out samples: {missing_predictions[:10]}"
                )
            scored_df = holdout_df.copy()
            scored_df["model_score"] = scored_df["sample_id"].map(score_dict)
            summary, winner_rank_df = compute_summary_metrics(
                scored_df,
                ranking_label_column=args.ranking_label_column,
                preference_gap=args.preference_gap,
                top_ks=top_ks,
                winner_count=args.winner_count,
            )
            summary_record = write_checkpoint_outputs(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                scored_df=scored_df,
                summary=summary,
                winner_rank_df=winner_rank_df,
            )
            summary_records.append(summary_record)
            log_checkpoint_to_wandb(
                wandb_run,
                summary_record=summary_record,
                checkpoint_index=checkpoint_index,
            )

        pd.DataFrame(summary_records).to_csv(output_dir / "benchmark_summary.csv", index=False)
        log_summary_table_to_wandb(wandb_run, wandb_module, summary_records)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
