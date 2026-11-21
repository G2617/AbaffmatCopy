# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import json
import math
import pickle

import pandas as pd
import pytest

from prepare_yeast_display_data import infer_parental_sequences, main as prepare_main
from datasets.preference_utils import (
    build_preference_pairs,
    parse_group_pair_cap_overrides,
)
from datasets.yeast_display import (
    assign_sample_splits,
    build_regions_from_sequence,
    canonicalize_yeast_display_dataframe,
    infer_chain_regions,
    parse_sort_population,
    save_canonicalization_state,
    write_json_samples,
)


PARENT_HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGGMDVWGQGTTVTVSS"
PARENT_LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLAWYQQKPGKAPKLLIYDASTRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPLTFGAGTKLELK"


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


HEAVY_REGIONS = {
    PARENT_HEAVY: {
        "imgt": {
            "fr1": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "cdr1": "GFTFSSYA",
            "fr2": "MSWVRQAPGKGLEWVS",
            "cdr2": "AISGSGGSTYYA",
            "fr3": "DSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
            "cdr3": "ARDRGGGMD",
            "fr4": "VWGQGTTVTVSS",
        }
    },
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARAAAAAADVWGQGTTVTVSS": {
        "imgt": {
            "fr1": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "cdr1": "GFTFSSYA",
            "fr2": "MSWVRQAPGKGLEWVS",
            "cdr2": "AISGSGGSTYYA",
            "fr3": "DSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
            "cdr3": "ARAAAAAAD",
            "fr4": "VWGQGTTVTVSS",
        }
    },
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGGMDVWGQGTTVTVSA": {
        "imgt": {
            "fr1": "EVQLVESGGGLVQPGGSLRLSCAAS",
            "cdr1": "GFTFSSYA",
            "fr2": "MSWVRQAPGKGLEWVS",
            "cdr2": "AISGSGGSTYYA",
            "fr3": "DSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
            "cdr3": "ARDRGGGMD",
            "fr4": "VWGQGTTVTVSA",
        }
    },
}

LIGHT_REGIONS = {
    PARENT_LIGHT: {
        "imgt": {
            "fr1": "DIQMTQSPSSLSASVGDRVTITCRAS",
            "cdr1": "QDISNYLA",
            "fr2": "WYQQKPGKAPKLLIY",
            "cdr2": "DASTRA",
            "fr3": "TGIPARFSGSGSGTDFTLTISSLEPEDFAVYYC",
            "cdr3": "QQYNSYP",
            "fr4": "LTFGAGTKLELK",
        },
        "kabat": {
            "fr1": "DIQMTQSPSSLSASVGDRVTITCRAS",
            "cdr1": "QDISNYLA",
            "fr2": "WYQQKPGKAPKLLI",
            "cdr2": "YDASTRA",
            "fr3": "TGIPARFSGSGSGTDFTLTISSLEPEDFAVYYC",
            "cdr3": "QQYNSYP",
            "fr4": "LTFGAGTKLELK",
        },
    }
}


def fake_regions(sequence: str, scheme: str = "imgt") -> dict[str, str]:
    if sequence in HEAVY_REGIONS:
        return HEAVY_REGIONS[sequence][scheme]
    return LIGHT_REGIONS[sequence][scheme]


def test_parse_sort_population():
    phase, mutated = parse_sort_population("phase1_h1_h2_am")
    assert phase == "phase1"
    assert mutated == ["H1", "H2"]


def test_infer_chain_regions_uses_kabat_for_light_l2(monkeypatch):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    light_regions = infer_chain_regions(PARENT_LIGHT, "light")
    assert light_regions["cdr1"] == "QDISNYLA"
    assert light_regions["cdr2"] == "YDASTRA"
    assert light_regions["cdr3"] == "QQYNSYP"
    assert light_regions == build_regions_from_sequence(
        PARENT_LIGHT,
        cdr1="QDISNYLA",
        cdr2="YDASTRA",
        cdr3="QQYNSYP",
    )


def test_canonicalize_yeast_display_dataframe(monkeypatch, tmp_path):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    raw_df = pd.DataFrame(
        [
            {
                "sort_population": "phase1_h3_am",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": 10,
            },
            {
                "sort_population": "phase1_h3_am",
                "sequence_aa_heavy": "EVQLX",
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": 2,
            },
            {
                "sort_population": "phase1_h1_h2_am",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "QQQQ",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": 3,
            },
        ]
    )
    canonical_df, rejected_df = canonicalize_yeast_display_dataframe(
        raw_df=raw_df,
        antigen_name="covid19_rbd",
        antigen_sequence="RBDSEQ",
        experimental_round="round1",
        parental_id="parentA",
        parental_heavy_sequence=PARENT_HEAVY,
        parental_light_sequence=PARENT_LIGHT,
        split_seed=1,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    assert len(canonical_df) == 1
    record = canonical_df.iloc[0].to_dict()
    assert record["phase"] == "phase1"
    assert json.loads(record["mutated_cdrs"]) == ["H3"]
    assert record["antigen_name"] == "covid19_rbd"
    assert record["cdr1_aa_heavy"] == "GFTFSSYA"
    assert record["cdr2_aa_light"] == "YDASTRA"
    assert record["cdr3_aa_light"] == "QQYNSYP"
    assert math.isclose(record["label_raw"], 1.0)
    assert record["label_std"] == 0.0
    assert set(rejected_df["reason"].tolist()) == {
        "heavy sequence contains noncanonical residues",
        "cdr3_aa_heavy value does not match IMGT-numbered region for sample",
    }

    write_json_samples(canonical_df, str(tmp_path / "json"), "RBDSEQ")
    sample_path = tmp_path / "json" / f"{record['sample_id']}.json"
    with open(sample_path, "r") as handle:
        payload = json.load(handle)
    assert payload[0]["name"] == record["sample_id"]
    with open(tmp_path / "json" / "id_labels.pkl", "rb") as handle:
        labels = pickle.load(handle)
    assert labels[record["sample_id"]] == 0.0


def test_canonicalize_trusts_fully_annotated_cdr_columns(monkeypatch):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    mutated_light = (
        "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLAWYQQKPGKAPKLLIYDASTRA"
        "TGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHYSAFPLTFGAGTKLELK"
    )
    raw_df = pd.DataFrame(
        [
            {
                "sort_population": "phase1_l3_am",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": mutated_light,
                "cdr1_aa_heavy": "GFTFSSYA",
                "cdr2_aa_heavy": "AISGSGGSTYYA",
                "cdr3_aa_heavy": "ARDRGGGMD",
                "cdr1_aa_light": "QDISNYLA",
                "cdr2_aa_light": "YDASTRA",
                "cdr3_aa_light": "QHYSAFP",
                "Redundancy": 4,
            },
        ]
    )

    canonical_df, rejected_df = canonicalize_yeast_display_dataframe(
        raw_df=raw_df,
        antigen_name="covid19_rbd",
        antigen_sequence="RBDSEQ",
        experimental_round="round1",
        parental_id="parentA",
        parental_heavy_sequence=PARENT_HEAVY,
        parental_light_sequence=PARENT_LIGHT,
    )

    assert rejected_df.empty
    assert len(canonical_df) == 1
    assert canonical_df.iloc[0]["cdr3_aa_light"] == "QHYSAFP"


def test_canonicalize_can_allow_offtarget_framework_mutations(monkeypatch):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    framework_mutated_heavy = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYA"
        "DSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGGMDVWGQGTTVTVSA"
    )
    raw_df = pd.DataFrame(
        [
            {
                "sort_population": "phase1_h3_am",
                "sequence_aa_heavy": framework_mutated_heavy,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": 4,
            },
        ]
    )

    strict_df, strict_rejected_df = canonicalize_yeast_display_dataframe(
        raw_df=raw_df,
        antigen_name="covid19_rbd",
        antigen_sequence="RBDSEQ",
        experimental_round="round1",
        parental_id="parentA",
        parental_heavy_sequence=PARENT_HEAVY,
        parental_light_sequence=PARENT_LIGHT,
    )
    relaxed_df, relaxed_rejected_df = canonicalize_yeast_display_dataframe(
        raw_df=raw_df,
        antigen_name="covid19_rbd",
        antigen_sequence="RBDSEQ",
        experimental_round="round1",
        parental_id="parentA",
        parental_heavy_sequence=PARENT_HEAVY,
        parental_light_sequence=PARENT_LIGHT,
        allow_offtarget_framework_mutations=True,
    )

    assert strict_df.empty
    assert strict_rejected_df.iloc[0]["reason"] == "heavy framework deviates from parental reference"
    assert relaxed_rejected_df.empty
    assert len(relaxed_df) == 1


def test_assign_sample_splits_stratifies_within_libraries():
    df = pd.DataFrame(
        {
            "library_name": ["large"] * 100 + ["small"] * 20,
            "sample_id": [f"id_{index}" for index in range(120)],
        }
    )
    split = assign_sample_splits(
        df,
        seed=7,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    df = df.assign(split=split)
    assert df["split"].value_counts().to_dict() == {
        "train": 96,
        "test": 12,
        "val": 12,
    }
    assert df.groupby(["library_name", "split"]).size().to_dict() == {
        ("large", "test"): 10,
        ("large", "train"): 80,
        ("large", "val"): 10,
        ("small", "test"): 2,
        ("small", "train"): 16,
        ("small", "val"): 2,
    }


def test_infer_parental_sequences_from_parental_sort_population():
    raw_df = pd.DataFrame(
        [
            {
                "sort_population": " parental ",
                "sequence_aa_heavy": f" {PARENT_HEAVY.lower()} ",
                "sequence_aa_light": f" {PARENT_LIGHT.lower()} ",
                "Redundancy": "-",
            },
            {
                "sort_population": "phase1_h3_am",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "Redundancy": 10,
            },
        ]
    )

    assert infer_parental_sequences(raw_df) == (PARENT_HEAVY, PARENT_LIGHT)


def test_prepare_main_infers_parent_and_skips_parental_row(monkeypatch, tmp_path):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "canonical.csv"
    json_dir = tmp_path / "json"
    audit_path = tmp_path / "rejected.csv"
    pd.DataFrame(
        [
            {
                "sort_population": "parental",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": "-",
            },
            {
                "sort_population": "phase1_h3_am",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": 10,
            },
        ]
    ).to_csv(raw_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_yeast_display_data.py",
            "--raw_yeast_csv_path",
            str(raw_path),
            "--canonical_output_path",
            str(output_path),
            "--json_output_dir",
            str(json_dir),
            "--audit_output_path",
            str(audit_path),
            "--antigen_sequence",
            "RBDSEQ",
            "--experimental_round",
            "round1",
        ],
    )

    prepare_main()

    canonical_df = pd.read_csv(output_path)
    assert len(canonical_df) == 1
    assert canonical_df.iloc[0]["parental_id"] == "parental"
    assert canonical_df.iloc[0]["redundancy"] == 10
    assert "parental" not in audit_path.read_text()


def test_canonicalize_can_resume_from_state(monkeypatch, tmp_path):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    mutated_heavy = (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYA"
        "DSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARAAAAAADVWGQGTTVTVSS"
    )
    raw_df = pd.DataFrame(
        [
            {
                "sort_population": "phase1_h3_am",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": 10,
            },
            {
                "sort_population": "phase1_h3_am",
                "sequence_aa_heavy": mutated_heavy,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "",
                "cdr2_aa_heavy": "",
                "cdr3_aa_heavy": "",
                "cdr1_aa_light": "",
                "cdr2_aa_light": "",
                "cdr3_aa_light": "",
                "Redundancy": 5,
            },
        ]
    )
    state_path = tmp_path / "canonicalization_state.pkl"
    save_canonicalization_state(
        str(state_path),
        next_position=1,
        raw_num_rows=len(raw_df),
        accepted_records=[],
        rejected_records=[],
    )

    canonical_df, rejected_df = canonicalize_yeast_display_dataframe(
        raw_df=raw_df,
        antigen_name="covid19_rbd",
        antigen_sequence="RBDSEQ",
        experimental_round="round1",
        parental_id="parentA",
        parental_heavy_sequence=PARENT_HEAVY,
        parental_light_sequence=PARENT_LIGHT,
        checkpoint_interval=1,
        checkpoint_state_path=str(state_path),
    )

    assert rejected_df.empty
    assert len(canonical_df) == 1
    assert canonical_df.iloc[0]["sequence_aa_heavy"] == mutated_heavy
    assert canonical_df.iloc[0]["redundancy"] == 5


def test_build_preference_pairs_hard_negative():
    canonical_df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "library_name": "lib1",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "phase": "phase1",
                "label_std": 2.0,
            },
            {
                "sample_id": "b",
                "library_name": "lib1",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "phase": "phase1",
                "label_std": 1.2,
            },
            {
                "sample_id": "c",
                "library_name": "lib1",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "phase": "phase1",
                "label_std": 0.1,
            },
        ]
    )
    pairs = build_preference_pairs(
        canonical_df,
        preference_gap=0.5,
        pair_sampling_strategy="hard_negative",
        max_pairs_per_group=2,
    )
    assert [(pair["chosen_id"], pair["rejected_id"]) for pair in pairs] == [("a", "b"), ("b", "c")]


def build_preference_pairs_exhaustive_for_test(
    canonical_df: pd.DataFrame,
    preference_gap: float,
    pair_sampling_strategy: str,
    max_pairs_per_group: int,
) -> list[tuple[str, str]]:
    pairs = []
    group_columns = ["library_name", "experimental_round", "antigen_name"]
    for _, group_df in canonical_df.groupby(group_columns):
        group_df = group_df.sort_values("label_std", ascending=False).reset_index(drop=True)
        candidate_lists = []
        for chosen_idx, chosen_row in group_df.iterrows():
            candidates = []
            for rejected_idx in range(chosen_idx + 1, len(group_df)):
                rejected_row = group_df.iloc[rejected_idx]
                gap = float(chosen_row["label_std"] - rejected_row["label_std"])
                if gap >= preference_gap:
                    candidates.append((chosen_row["sample_id"], rejected_row["sample_id"], gap))
            if pair_sampling_strategy == "hard_negative":
                candidates = sorted(candidates, key=lambda item: item[2])
            candidate_lists.append(candidates)
        group_pairs = []
        if pair_sampling_strategy == "all":
            for candidates in candidate_lists:
                group_pairs.extend(candidates)
            group_pairs = sorted(group_pairs, key=lambda item: item[2])
            if max_pairs_per_group > 0:
                group_pairs = group_pairs[:max_pairs_per_group]
        else:
            depth = 0
            while True:
                depth_pairs = [
                    candidates[depth]
                    for candidates in candidate_lists
                    if depth < len(candidates)
                ]
                if not depth_pairs:
                    break
                group_pairs.extend(depth_pairs)
                if max_pairs_per_group > 0 and len(group_pairs) >= max_pairs_per_group:
                    group_pairs = group_pairs[:max_pairs_per_group]
                    break
                depth += 1
        pairs.extend((chosen_id, rejected_id) for chosen_id, rejected_id, _ in group_pairs)
    return pairs


def test_build_preference_pairs_matches_exhaustive_order_for_capped_groups():
    canonical_df = pd.DataFrame(
        [
            {
                "sample_id": f"id_{index}",
                "library_name": "lib1",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "phase": "phase1",
                "label_std": label,
            }
            for index, label in enumerate([3.0, 2.7, 2.2, 1.8, 1.1, 0.7, 0.0])
        ]
    )

    for strategy in ("hard_negative", "all"):
        pairs = build_preference_pairs(
            canonical_df,
            preference_gap=0.5,
            pair_sampling_strategy=strategy,
            max_pairs_per_group=5,
        )
        assert [(pair["chosen_id"], pair["rejected_id"]) for pair in pairs] == (
            build_preference_pairs_exhaustive_for_test(
                canonical_df,
                preference_gap=0.5,
                pair_sampling_strategy=strategy,
                max_pairs_per_group=5,
            )
        )


def test_parse_group_pair_cap_overrides_accepts_cli_string():
    assert parse_group_pair_cap_overrides("lib_a=10, lib_b|r1|covid=4") == {
        "lib_a": 10,
        "lib_b|r1|covid": 4,
    }


def test_build_preference_pairs_respects_per_group_cap_overrides():
    canonical_df = pd.DataFrame(
        [
            {
                "sample_id": f"lib1_{index}",
                "library_name": "lib1",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "phase": "phase1",
                "label_std": label,
            }
            for index, label in enumerate([4.0, 3.0, 2.0, 1.0, 0.0])
        ]
        + [
            {
                "sample_id": f"lib2_{index}",
                "library_name": "lib2",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "phase": "phase1",
                "label_std": label,
            }
            for index, label in enumerate([5.0, 4.0, 3.0, 2.0, 1.0])
        ]
    )

    pairs = build_preference_pairs(
        canonical_df,
        preference_gap=0.5,
        pair_sampling_strategy="hard_negative",
        max_pairs_per_group=4,
        max_pairs_per_group_by_key={"lib1": 2, "lib2|r1|covid19_rbd": 3},
    )

    lib1_pairs = [pair for pair in pairs if pair["library_name"] == "lib1"]
    lib2_pairs = [pair for pair in pairs if pair["library_name"] == "lib2"]

    assert len(lib1_pairs) == 2
    assert len(lib2_pairs) == 3


def test_preference_dataset_can_build_samples_from_canonical_table(tmp_path):
    pytest.importorskip("torch")
    from datasets.preference_dataset import PreferenceDataset

    canonical_path = tmp_path / "canonical.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "high",
                "library_name": "lib1",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "antigen_sequence": "RBDSEQ",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "phase": "phase1",
                "label_std": 1.0,
                "split": "train",
            },
            {
                "sample_id": "low",
                "library_name": "lib1",
                "experimental_round": "r1",
                "antigen_name": "covid19_rbd",
                "antigen_sequence": "RBDSEQ",
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "phase": "phase1",
                "label_std": 0.0,
                "split": "train",
            },
        ]
    ).to_csv(canonical_path, index=False)
    configs = AttrDict(
        {
            "dump_dir": str(tmp_path / "dump"),
            "use_msa": False,
            "train_crop_size": -1,
            "max_samples": -1,
            "run_name": "unit",
            "antigen_sequence": "",
            "esm": AttrDict({"enable": False, "model_name": "esm2-3b"}),
        }
    )

    dataset = PreferenceDataset(
        canonical_path=str(canonical_path),
        json_output_dir="",
        split="train",
        configs=configs,
        preference_gap=0.5,
        max_pairs_per_group=10,
    )

    assert len(dataset.base_dataset.all_inputs) == 2
    assert dataset.base_dataset.all_inputs[0]["name"] == "high"
    assert dataset.base_dataset.all_inputs[0]["sequences"][0]["proteinChain"]["sequence"] == "RBDSEQ"
    assert len(dataset) == 1
    assert dataset.pairs[0]["chosen_id"] == "high"
    assert dataset.pairs[0]["rejected_id"] == "low"
