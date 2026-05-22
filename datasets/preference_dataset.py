# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

from typing import Any

from torch.utils.data import Dataset

from datasets.dataset import DistilledDataset
from datasets.preference_utils import (
    build_preference_pairs,
    parse_group_pair_cap_overrides,
)
from datasets.yeast_display import load_dataframe
from utils.torch_utils import collate_padding_speed_fn


def load_split_ids(split_path: str) -> set[str]:
    if not split_path:
        return set()
    with open(split_path, "r") as handle:
        return {line.strip() for line in handle if line.strip()}


def row_to_distilled_input(row: Any, antigen_sequence_override: str = "") -> dict[str, Any]:
    antigen_sequence = antigen_sequence_override or str(row["antigen_sequence"]).strip()
    return {
        "name": row["sample_id"],
        "sequences": [
            {"proteinChain": {"sequence": antigen_sequence, "count": 1}},
            {"proteinChain": {"sequence": row["sequence_aa_heavy"], "count": 1}},
            {"proteinChain": {"sequence": row["sequence_aa_light"], "count": 1}},
        ],
    }


class PreferenceDataset(Dataset):
    def __init__(
        self,
        canonical_path: str,
        json_output_dir: str,
        split: str,
        configs: Any,
        split_path: str = "",
        preference_gap: float = 0.5,
        pair_sampling_strategy: str = "hard_negative",
        max_pairs_per_group: int = 128,
    ) -> None:
        self.canonical_df = load_dataframe(canonical_path)
        split_ids = load_split_ids(split_path)
        if split_ids:
            self.canonical_df = self.canonical_df[
                self.canonical_df["sample_id"].isin(split_ids)
            ].copy()
        else:
            self.canonical_df = self.canonical_df[self.canonical_df["split"] == split].copy()
        self.canonical_df = self.canonical_df.reset_index(drop=True)
        crop_size = -1 if split != "train" else configs.train_crop_size
        if json_output_dir:
            self.base_dataset = DistilledDataset(
                input_json_path=json_output_dir,
                configs=configs,
                dump_dir=configs.dump_dir,
                use_msa=configs.use_msa,
                crop_size=crop_size,
            )
        else:
            input_samples = [
                row_to_distilled_input(
                    row,
                    antigen_sequence_override=getattr(configs, "antigen_sequence", ""),
                )
                for row in self.canonical_df.to_dict("records")
            ]
            self.base_dataset = DistilledDataset(
                input_json_path="",
                configs=configs,
                dump_dir=configs.dump_dir,
                use_msa=configs.use_msa,
                crop_size=crop_size,
                input_samples=input_samples,
                input_name=f"{split}_canonical",
            )
        self.row_by_sample_id = {
            row["sample_id"]: row for _, row in self.canonical_df.iterrows()
        }
        self.index_by_sample_id = {
            sample["name"]: index for index, sample in enumerate(self.base_dataset.all_inputs)
        }
        pair_cap_overrides = {}
        if split == "train":
            pair_cap_overrides = parse_group_pair_cap_overrides(
                getattr(configs, "max_pairs_per_group_by_key", "")
            )
        pairs = build_preference_pairs(
            self.canonical_df,
            preference_gap=preference_gap,
            pair_sampling_strategy=pair_sampling_strategy,
            max_pairs_per_group=max_pairs_per_group,
            max_pairs_per_group_by_key=pair_cap_overrides,
        )
        self.pairs = [
            pair
            for pair in pairs
            if pair["chosen_id"] in self.index_by_sample_id
            and pair["rejected_id"] in self.index_by_sample_id
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        chosen = self.base_dataset[self.index_by_sample_id[pair["chosen_id"]]]
        rejected = self.base_dataset[self.index_by_sample_id[pair["rejected_id"]]]
        chosen["sample_label"] = self.row_by_sample_id[pair["chosen_id"]]["label_std"]
        rejected["sample_label"] = self.row_by_sample_id[pair["rejected_id"]]["label_std"]
        return {
            "chosen": chosen,
            "rejected": rejected,
            "pair_metadata": pair,
        }


def collate_preference_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    chosen_examples = [example["chosen"] for example in batch]
    rejected_examples = [example["rejected"] for example in batch]
    pair_metadata = [example["pair_metadata"] for example in batch]
    return {
        "chosen_batch": collate_padding_speed_fn(chosen_examples),
        "rejected_batch": collate_padding_speed_fn(rejected_examples),
        "pair_metadata": pair_metadata,
    }
