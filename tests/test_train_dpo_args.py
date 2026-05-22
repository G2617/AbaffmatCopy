# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import pytest
from types import SimpleNamespace


def test_normalize_optional_empty_dpo_args():
    pytest.importorskip("torch")
    from runners.train_dpo import normalize_optional_empty_args

    assert normalize_optional_empty_args(
        [
            "--canonical_output_path",
            "canonical.csv",
            "--load_checkpoint_path",
            "--max_steps",
            "1",
        ]
    ) == [
        "--canonical_output_path",
        "canonical.csv",
        "--load_checkpoint_path",
        "__ABAFFMAT_EMPTY__",
        "--max_steps",
        "1",
    ]
    assert normalize_optional_empty_args(
        ["--load_checkpoint_path", "best.pt", "--reference_checkpoint_path"]
    ) == [
        "--load_checkpoint_path",
        "best.pt",
        "--reference_checkpoint_path",
        "__ABAFFMAT_EMPTY__",
    ]
    assert normalize_optional_empty_args(["--load_checkpoint_path", ""]) == [
        "--load_checkpoint_path",
        "__ABAFFMAT_EMPTY__",
    ]


def test_dpo_accumulation_helpers():
    pytest.importorskip("torch")
    from runners.train_dpo import get_accumulation_steps, should_step_optimizer

    assert get_accumulation_steps(SimpleNamespace(iters_to_accumulate=4)) == 4
    assert get_accumulation_steps(SimpleNamespace(iters_to_accumulate=0)) == 1
    assert should_step_optimizer(micro_step=3, accumulation_steps=4) is False
    assert should_step_optimizer(micro_step=4, accumulation_steps=4) is True


def test_grad_clip_norm_accepts_float_cli_value():
    pytest.importorskip("ml_collections")
    from configs.config import parse_configs
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs

    configs = parse_configs(
        configs={**configs_base, **{"data": data_configs}, **inference_configs},
        arg_str=(
            "--project test --run_name test --base_dir outputs "
            "--eval_interval -1 --log_interval 1 --max_steps 1 "
            "--input_json_path dummy.json --grad_clip_norm 1.0"
        ),
        fill_required_with_null=True,
    )

    assert configs.grad_clip_norm == pytest.approx(1.0)


def test_dpo_build_optimizer_honors_adamw_config():
    torch = pytest.importorskip("torch")
    from runners.train_dpo import build_optimizer

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    configs = SimpleNamespace(
        lr=3e-5,
        adam=SimpleNamespace(
            beta1=0.8,
            beta2=0.9,
            weight_decay=1e-6,
            use_adamw=True,
        ),
    )

    optimizer = build_optimizer([parameter], configs)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-6)


def test_group_metric_helpers_aggregate_per_group():
    torch = pytest.importorskip("torch")
    from runners.train_dpo import (
        average_metric_sums,
        preference_group_label,
        sanitize_metric_component,
        update_group_metric_sums,
    )

    pair_metadata = [
        {
            "library_name": "phase1_l1_l2_am",
            "experimental_round": "round1",
            "antigen_name": "covid19_rbd",
        },
        {
            "library_name": "phase1_l1_l2_am",
            "experimental_round": "round1",
            "antigen_name": "covid19_rbd",
        },
        {
            "library_name": "phase1_l3_am",
            "experimental_round": "round1",
            "antigen_name": "covid19_rbd",
        },
    ]
    example_metrics = {
        "loss": torch.tensor([0.2, 0.4, 0.6]),
        "pair_accuracy": torch.tensor([1.0, 0.0, 1.0]),
        "policy_margin": torch.tensor([0.8, -0.2, 0.3]),
    }

    group_metric_sums = {}
    group_pair_counts = {}
    update_group_metric_sums(
        group_metric_sums,
        group_pair_counts,
        pair_metadata,
        example_metrics,
    )

    lib1_key = preference_group_label(pair_metadata[0])
    lib3_key = preference_group_label(pair_metadata[2])
    assert sanitize_metric_component(lib1_key) == "phase1_l1_l2_am_round1_covid19_rbd"
    assert group_pair_counts == {lib1_key: 2, lib3_key: 1}
    assert average_metric_sums(group_metric_sums[lib1_key], group_pair_counts[lib1_key]) == {
        "loss": pytest.approx(0.3),
        "pair_accuracy": pytest.approx(0.5),
        "policy_margin": pytest.approx(0.3),
    }
    assert average_metric_sums(group_metric_sums[lib3_key], group_pair_counts[lib3_key]) == {
        "loss": pytest.approx(0.6),
        "pair_accuracy": pytest.approx(1.0),
        "policy_margin": pytest.approx(0.3),
    }
