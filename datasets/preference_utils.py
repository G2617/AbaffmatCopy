# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

from bisect import bisect_left
import heapq
from typing import Any

import pandas as pd


def make_preference_pair(
    group_key: tuple[Any, Any, Any],
    chosen_row: pd.Series,
    rejected_row: pd.Series,
    gap: float,
) -> dict[str, Any]:
    return {
        "chosen_id": chosen_row["sample_id"],
        "rejected_id": rejected_row["sample_id"],
        "chosen_label": float(chosen_row["label_std"]),
        "rejected_label": float(rejected_row["label_std"]),
        "preference_gap": gap,
        "library_name": group_key[0],
        "experimental_round": group_key[1],
        "antigen_name": group_key[2],
        "phase": chosen_row["phase"],
    }


def parse_group_pair_cap_overrides(raw_value: Any) -> dict[str, int]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return {
            str(key).strip(): int(value)
            for key, value in raw_value.items()
            if str(key).strip()
        }
    text = str(raw_value).strip()
    if not text:
        return {}
    overrides = {}
    for entry in text.split(","):
        item = entry.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "group pair-cap overrides must use 'key=value' entries separated by commas"
            )
        key, value = item.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("group pair-cap override keys must be non-empty")
        overrides[normalized_key] = int(value.strip())
    return overrides


def resolve_group_pair_cap(
    group_key: tuple[Any, Any, Any],
    default_cap: int,
    overrides: dict[str, int] | None = None,
) -> int:
    if not overrides:
        return default_cap
    library_name, experimental_round, antigen_name = group_key
    composite_key = f"{library_name}|{experimental_round}|{antigen_name}"
    for candidate_key in (
        composite_key,
        str(library_name),
    ):
        if candidate_key in overrides:
            return int(overrides[candidate_key])
    return default_cap


def get_first_rejected_indices(
    labels_desc: list[float],
    preference_gap: float,
) -> list[int]:
    # labels_desc is sorted high-to-low. Negating gives an ascending list for bisect.
    neg_labels = [-label for label in labels_desc]
    first_indices = []
    for chosen_idx, chosen_label in enumerate(labels_desc):
        max_rejected_label = chosen_label - preference_gap
        first_idx = bisect_left(
            neg_labels,
            -max_rejected_label,
            lo=chosen_idx + 1,
        )
        first_indices.append(first_idx)
    return first_indices


def build_hard_negative_pairs_for_group(
    group_key: tuple[Any, Any, Any],
    group_df: pd.DataFrame,
    preference_gap: float,
    max_pairs_per_group: int,
) -> list[dict[str, Any]]:
    labels = [float(value) for value in group_df["label_std"].tolist()]
    first_rejected_indices = get_first_rejected_indices(labels, preference_gap)
    pairs = []
    depth = 0
    n_rows = len(group_df)
    while True:
        depth_pairs = []
        for chosen_idx, first_rejected_idx in enumerate(first_rejected_indices):
            rejected_idx = first_rejected_idx + depth
            if rejected_idx >= n_rows:
                continue
            chosen_row = group_df.iloc[chosen_idx]
            rejected_row = group_df.iloc[rejected_idx]
            depth_pairs.append(
                make_preference_pair(
                    group_key,
                    chosen_row,
                    rejected_row,
                    float(labels[chosen_idx] - labels[rejected_idx]),
                )
            )
        if not depth_pairs:
            break
        pairs.extend(depth_pairs)
        if max_pairs_per_group > 0 and len(pairs) >= max_pairs_per_group:
            return pairs[:max_pairs_per_group]
        depth += 1
    return pairs


def build_all_pairs_for_group(
    group_key: tuple[Any, Any, Any],
    group_df: pd.DataFrame,
    preference_gap: float,
    max_pairs_per_group: int,
) -> list[dict[str, Any]]:
    labels = [float(value) for value in group_df["label_std"].tolist()]
    first_rejected_indices = get_first_rejected_indices(labels, preference_gap)
    if max_pairs_per_group > 0:
        heap = []
        for chosen_idx, first_rejected_idx in enumerate(first_rejected_indices):
            if first_rejected_idx < len(group_df):
                heapq.heappush(
                    heap,
                    (
                        labels[chosen_idx] - labels[first_rejected_idx],
                        chosen_idx,
                        first_rejected_idx,
                    ),
                )
        scored_indices = []
        while heap and len(scored_indices) < max_pairs_per_group:
            gap, chosen_idx, rejected_idx = heapq.heappop(heap)
            scored_indices.append((gap, chosen_idx, rejected_idx))
            next_rejected_idx = rejected_idx + 1
            if next_rejected_idx < len(group_df):
                heapq.heappush(
                    heap,
                    (
                        labels[chosen_idx] - labels[next_rejected_idx],
                        chosen_idx,
                        next_rejected_idx,
                    ),
                )
        return [
            make_preference_pair(
                group_key,
                group_df.iloc[chosen_idx],
                group_df.iloc[rejected_idx],
                float(gap),
            )
            for gap, chosen_idx, rejected_idx in scored_indices
        ]

    scored_indices = []
    for chosen_idx, first_rejected_idx in enumerate(first_rejected_indices):
        for rejected_idx in range(first_rejected_idx, len(group_df)):
            scored_indices.append(
                (labels[chosen_idx] - labels[rejected_idx], chosen_idx, rejected_idx)
            )
    scored_indices.sort(key=lambda item: item[0])
    return [
        make_preference_pair(
            group_key,
            group_df.iloc[chosen_idx],
            group_df.iloc[rejected_idx],
            float(gap),
        )
        for gap, chosen_idx, rejected_idx in scored_indices
    ]


def build_preference_pairs(
    canonical_df: pd.DataFrame,
    preference_gap: float = 0.5,
    pair_sampling_strategy: str = "hard_negative",
    max_pairs_per_group: int = 128,
    max_pairs_per_group_by_key: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    pairs = []
    group_columns = ["library_name", "experimental_round", "antigen_name"]
    for group_key, group_df in canonical_df.groupby(group_columns):
        group_df = group_df.sort_values("label_std", ascending=False).reset_index(drop=True)
        group_cap = resolve_group_pair_cap(
            group_key,
            default_cap=max_pairs_per_group,
            overrides=max_pairs_per_group_by_key,
        )
        if pair_sampling_strategy == "all":
            group_pairs = build_all_pairs_for_group(
                group_key,
                group_df,
                preference_gap,
                group_cap,
            )
        else:
            group_pairs = build_hard_negative_pairs_for_group(
                group_key,
                group_df,
                preference_gap,
                group_cap,
            )
        pairs.extend(group_pairs)
    return pairs
