# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_preference_loss_metrics(
    loss_type: str,
    beta: float,
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor | None = None,
    reference_rejected: torch.Tensor | None = None,
    return_example_metrics: bool = False,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor] | None,
]:
    normalized_loss_type = loss_type.strip().lower()
    policy_chosen = policy_chosen.reshape(-1)
    policy_rejected = policy_rejected.reshape(-1)
    policy_margin = policy_chosen - policy_rejected

    if normalized_loss_type in {"anchored_dpo", "dpo"}:
        if reference_chosen is None or reference_rejected is None:
            raise ValueError("anchored_dpo requires reference scores")
        reference_chosen = reference_chosen.reshape(-1)
        reference_rejected = reference_rejected.reshape(-1)
        reference_margin = reference_chosen - reference_rejected
        relative_margin = policy_margin - reference_margin
        logits = beta * relative_margin
        loss = -F.logsigmoid(logits)
        example_metrics = {
            "loss": loss.detach(),
            "policy_chosen_mean": policy_chosen.detach(),
            "policy_rejected_mean": policy_rejected.detach(),
            "reference_chosen_mean": reference_chosen.detach(),
            "reference_rejected_mean": reference_rejected.detach(),
            "policy_margin": policy_margin.detach(),
            "reference_margin": reference_margin.detach(),
            "relative_margin": relative_margin.detach(),
            "dpo_logit": logits.detach(),
            "policy_pair_accuracy": (policy_margin > 0).float(),
            "reference_pair_accuracy": (reference_margin > 0).float(),
            "relative_margin_accuracy": (relative_margin > 0).float(),
        }
    elif normalized_loss_type == "pairwise_logistic":
        logits = beta * policy_margin
        loss = -F.logsigmoid(logits)
        zero_vec = policy_margin.detach().new_zeros(policy_margin.shape)
        example_metrics = {
            "loss": loss.detach(),
            "policy_chosen_mean": policy_chosen.detach(),
            "policy_rejected_mean": policy_rejected.detach(),
            "reference_chosen_mean": zero_vec,
            "reference_rejected_mean": zero_vec,
            "policy_margin": policy_margin.detach(),
            "reference_margin": zero_vec,
            "relative_margin": policy_margin.detach(),
            "dpo_logit": logits.detach(),
            "policy_pair_accuracy": (policy_margin > 0).float(),
            "reference_pair_accuracy": zero_vec,
            "relative_margin_accuracy": (policy_margin > 0).float(),
        }
    else:
        raise ValueError(
            "Unsupported dpo_loss_type "
            + f"{loss_type!r}; expected 'anchored_dpo' or 'pairwise_logistic'"
        )

    metrics = {key: value.mean() for key, value in example_metrics.items()}
    metrics["pair_accuracy"] = metrics["policy_pair_accuracy"]
    if return_example_metrics:
        example_metrics["pair_accuracy"] = example_metrics["policy_pair_accuracy"]
        return loss.mean(), metrics, example_metrics
    return loss.mean(), metrics, None


class AnchoredDPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen: torch.Tensor,
        policy_rejected: torch.Tensor,
        reference_chosen: torch.Tensor,
        reference_rejected: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics, _ = compute_preference_loss_metrics(
            loss_type="anchored_dpo",
            beta=self.beta,
            policy_chosen=policy_chosen,
            policy_rejected=policy_rejected,
            reference_chosen=reference_chosen,
            reference_rejected=reference_rejected,
            return_example_metrics=False,
        )
        return loss, metrics


class PairwiseLogisticLoss(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen: torch.Tensor,
        policy_rejected: torch.Tensor,
        reference_chosen: torch.Tensor | None = None,
        reference_rejected: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics, _ = compute_preference_loss_metrics(
            loss_type="pairwise_logistic",
            beta=self.beta,
            policy_chosen=policy_chosen,
            policy_rejected=policy_rejected,
            reference_chosen=reference_chosen,
            reference_rejected=reference_rejected,
            return_example_metrics=False,
        )
        return loss, metrics


def build_preference_loss(loss_type: str, beta: float) -> nn.Module:
    normalized_loss_type = loss_type.strip().lower()
    if normalized_loss_type in {"anchored_dpo", "dpo"}:
        return AnchoredDPOLoss(beta=beta)
    if normalized_loss_type == "pairwise_logistic":
        return PairwiseLogisticLoss(beta=beta)
    raise ValueError(
        "Unsupported dpo_loss_type "
        + f"{loss_type!r}; expected 'anchored_dpo' or 'pairwise_logistic'"
    )
