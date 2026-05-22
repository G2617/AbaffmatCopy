# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import pytest

torch = pytest.importorskip("torch")
import torch

from losses.dpo import (
    AnchoredDPOLoss,
    PairwiseLogisticLoss,
    build_preference_loss,
    compute_preference_loss_metrics,
)


def test_anchored_dpo_loss_smoke():
    loss_fn = AnchoredDPOLoss(beta=0.1)
    policy_chosen = torch.tensor([1.5, 1.0])
    policy_rejected = torch.tensor([0.5, 0.25])
    reference_chosen = torch.tensor([1.0, 0.8])
    reference_rejected = torch.tensor([0.6, 0.4])
    loss, metrics = loss_fn(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
    )
    assert torch.isfinite(loss)
    assert metrics["pair_accuracy"].item() == 1.0
    assert metrics["policy_pair_accuracy"].item() == 1.0
    assert metrics["reference_pair_accuracy"].item() == 1.0
    assert metrics["relative_margin_accuracy"].item() == 1.0
    assert metrics["policy_margin"].item() > 0
    assert metrics["relative_margin"].item() > 0
    assert metrics["policy_chosen_mean"].item() == pytest.approx(1.25)
    assert metrics["policy_rejected_mean"].item() == pytest.approx(0.375)


def test_anchored_dpo_loss_tracks_relative_margin_accuracy():
    loss_fn = AnchoredDPOLoss(beta=0.1)
    policy_chosen = torch.tensor([1.0, 2.0])
    policy_rejected = torch.tensor([0.0, 1.5])
    reference_chosen = torch.tensor([0.0, 4.0])
    reference_rejected = torch.tensor([0.0, 2.0])

    _, metrics = loss_fn(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
    )

    assert metrics["policy_pair_accuracy"].item() == 1.0
    assert metrics["reference_pair_accuracy"].item() == 0.5
    assert metrics["relative_margin_accuracy"].item() == 0.5
    assert metrics["policy_margin"].item() == pytest.approx(0.75)
    assert metrics["reference_margin"].item() == pytest.approx(1.0)
    assert metrics["relative_margin"].item() == pytest.approx(-0.25)


def test_pairwise_logistic_loss_uses_policy_margin_directly():
    loss_fn = PairwiseLogisticLoss(beta=1.0)
    policy_chosen = torch.tensor([1.0, 2.0])
    policy_rejected = torch.tensor([0.0, 2.5])

    loss, metrics = loss_fn(policy_chosen, policy_rejected)

    expected_loss = -torch.nn.functional.logsigmoid(
        policy_chosen - policy_rejected
    ).mean()
    assert loss.item() == pytest.approx(expected_loss.item())
    assert metrics["policy_margin"].item() == pytest.approx(0.25)
    assert metrics["relative_margin"].item() == pytest.approx(0.25)
    assert metrics["policy_pair_accuracy"].item() == pytest.approx(0.5)
    assert metrics["relative_margin_accuracy"].item() == pytest.approx(0.5)
    assert metrics["reference_margin"].item() == 0.0


def test_build_preference_loss_factory():
    assert isinstance(build_preference_loss("anchored_dpo", beta=0.1), AnchoredDPOLoss)
    assert isinstance(
        build_preference_loss("pairwise_logistic", beta=1.0),
        PairwiseLogisticLoss,
    )
    with pytest.raises(ValueError):
        build_preference_loss("unknown", beta=1.0)


def test_compute_preference_loss_metrics_can_return_example_metrics():
    loss, metrics, example_metrics = compute_preference_loss_metrics(
        loss_type="pairwise_logistic",
        beta=1.0,
        policy_chosen=torch.tensor([1.0, 2.0]),
        policy_rejected=torch.tensor([0.0, 2.5]),
        return_example_metrics=True,
    )

    assert torch.isfinite(loss)
    assert metrics["pair_accuracy"].item() == pytest.approx(0.5)
    assert example_metrics is not None
    assert example_metrics["loss"].shape == (2,)
    assert example_metrics["pair_accuracy"].tolist() == pytest.approx([1.0, 0.0])
