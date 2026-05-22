# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import json

import pandas as pd
import pytest

from benchmark_yeast_display import (
    compute_summary_metrics,
    checkpoint_label,
    load_holdout_dataframe,
    normalize_score_dict,
    parse_bool_arg,
    parse_top_ks,
    wandb_metric_payload,
    write_holdout_score_inputs,
)


def make_record(sample_id: str, label_raw: float, label_std: float, model_score: float) -> dict:
    return {
        "sample_id": sample_id,
        "library_name": "lib1",
        "experimental_round": "round1",
        "antigen_name": "covid19_rbd",
        "phase": "phase1",
        "label_raw": label_raw,
        "label_std": label_std,
        "model_score": model_score,
    }


def make_canonical_record(sample_id: str, label_raw: float, label_std: float) -> dict:
    record = make_record(sample_id, label_raw, label_std, model_score=0.0)
    record.update(
        {
            "antigen_sequence": "ACDE",
            "sequence_aa_heavy": "FGHI",
            "sequence_aa_light": "KLMN",
            "split": "test",
        }
    )
    return record


def test_parse_top_ks_deduplicates():
    assert parse_top_ks("1,5,5,10") == [1, 5, 10]


def test_parse_bool_arg_matches_config_bool_values():
    assert parse_bool_arg("True") is True
    assert parse_bool_arg("yes") is True
    assert parse_bool_arg("False") is False
    assert parse_bool_arg("0") is False


def test_normalize_score_dict_flattens_scalars():
    normalized = normalize_score_dict({"a": [1.25], "b": 0.5})
    assert normalized == {"a": 1.25, "b": 0.5}


def test_checkpoint_label_handles_base_checkpoint():
    assert checkpoint_label("") == "base"
    assert checkpoint_label("outputs/run/checkpoints/best.pt") == "best"


def test_wandb_metric_payload_keeps_only_finite_numeric_metrics():
    payload = wandb_metric_payload(
        {
            "checkpoint_path": "",
            "run_dir": "outputs/run",
            "pairwise_accuracy": 0.75,
            "pair_count": 4,
            "spearman_label_raw": None,
            "bad_metric": float("nan"),
            "label": "base",
        }
    )

    assert payload == {
        "benchmark/pairwise_accuracy": 0.75,
        "benchmark/pair_count": 4.0,
    }


def test_compute_summary_metrics_reports_pairwise_and_topk():
    scored_df = pd.DataFrame(
        [
            make_record("a", label_raw=3.0, label_std=2.0, model_score=0.9),
            make_record("b", label_raw=2.0, label_std=1.0, model_score=0.6),
            make_record("c", label_raw=1.0, label_std=0.0, model_score=0.1),
        ]
    )

    summary, winner_rank_df = compute_summary_metrics(
        scored_df,
        ranking_label_column="label_raw",
        preference_gap=0.5,
        top_ks=[1, 2],
        winner_count=2,
    )

    assert summary["pair_count"] == 3
    assert summary["pairwise_accuracy"] == 1.0
    assert summary["spearman_label_raw"] == 1.0
    assert summary["top_1_overlap"] == 1.0
    assert summary["top_2_overlap"] == 2.0
    assert summary["best_winner_rank"] == 1.0
    assert winner_rank_df["model_rank"].tolist() == [1, 2]


def test_load_holdout_dataframe_does_not_require_json_dir(tmp_path):
    canonical_path = tmp_path / "canonical.csv"
    holdout_path = tmp_path / "holdout.txt"
    pd.DataFrame(
        [
            make_canonical_record("drop", label_raw=0.0, label_std=0.0),
            make_canonical_record("keep_b", label_raw=1.0, label_std=1.0),
            make_canonical_record("keep_a", label_raw=2.0, label_std=2.0),
        ]
    ).to_csv(canonical_path, index=False)
    holdout_path.write_text("keep_a\nkeep_b\n")

    holdout_df, ordered_ids = load_holdout_dataframe(
        canonical_output_path=str(canonical_path),
        holdout_ids_path=str(holdout_path),
    )

    assert ordered_ids == ["keep_a", "keep_b"]
    assert holdout_df["sample_id"].tolist() == ["keep_a", "keep_b"]


def test_write_holdout_score_inputs_from_canonical_rows(tmp_path):
    holdout_df = pd.DataFrame(
        [
            make_canonical_record("sample_a", label_raw=2.0, label_std=2.0),
            make_canonical_record("sample_b", label_raw=1.0, label_std=1.0),
        ]
    )

    json_dir = write_holdout_score_inputs(holdout_df, tmp_path)

    sample_path = tmp_path / "canonical_holdout_json" / "sample_a.json"
    assert json_dir == str(tmp_path / "canonical_holdout_json")
    payload = json.loads(sample_path.read_text())
    assert payload == [
        {
            "name": "sample_a",
            "sequences": [
                {"proteinChain": {"sequence": "ACDE", "count": 1}},
                {"proteinChain": {"sequence": "FGHI", "count": 1}},
                {"proteinChain": {"sequence": "KLMN", "count": 1}},
            ],
        }
    ]


def test_distilled_dataset_filters_to_requested_sample_ids(tmp_path):
    pytest.importorskip("biotite")
    from datasets.dataset import DistilledDataset

    sample_dir = tmp_path / "json"
    sample_dir.mkdir()
    (sample_dir / "keep.json").write_text(
        json.dumps(
            [
                {
                    "name": "keep",
                    "sequences": [{"proteinChain": {"sequence": "AAAA", "count": 1}}],
                }
            ]
        )
    )
    (sample_dir / "drop.json").write_text(
        json.dumps(
            [
                {
                    "name": "drop",
                    "sequences": [{"proteinChain": {"sequence": "BBBB", "count": 1}}],
                }
            ]
        )
    )
    (sample_dir / "id_labels.pkl").write_bytes(b"\x80\x04\x95\x1f\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x04keep\x94G?\xf0\x00\x00\x00\x00\x00\x00\x8c\x04drop\x94G@\x00\x00\x00\x00\x00\x00\x00u.")
    id_path = tmp_path / "holdout_ids.txt"
    id_path.write_text("keep\n")

    dataset = DistilledDataset(
        input_json_path=str(sample_dir),
        dump_dir=str(tmp_path / "dump"),
        configs=None,
        use_msa=False,
        crop_size=-1,
        sample_ids_path=str(id_path),
    )

    assert len(dataset) == 1
    assert dataset.all_inputs[0]["name"] == "keep"
    assert dataset.all_labels == [1.0]
