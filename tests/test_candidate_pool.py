# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import json

import pandas as pd

from sample_candidate_pool import parse_target_cdrs, sample_candidate_records


PARENT_HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDRGGGMDVWGQGTTVTVSS"
PARENT_LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQDISNYLAWYQQKPGKAPKLLIYDASTRATGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPLTFGAGTKLELK"


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
    }
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


def make_heavy(cdr1: str, cdr2: str, cdr3: str) -> str:
    return (
        "EVQLVESGGGLVQPGGSLRLSCAAS"
        + cdr1
        + "MSWVRQAPGKGLEWVS"
        + cdr2
        + "DSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC"
        + cdr3
        + "VWGQGTTVTVSS"
    )


def make_light(cdr1: str, cdr2: str, cdr3: str) -> str:
    return (
        "DIQMTQSPSSLSASVGDRVTITCRAS"
        + cdr1
        + "WYQQKPGKAPKLLI"
        + cdr2
        + "TGIPARFSGSGSGTDFTLTISSLEPEDFAVYYC"
        + cdr3
        + "LTFGAGTKLELK"
    )


def test_parse_target_cdrs_deduplicates_and_validates():
    assert parse_target_cdrs("h1, H2, h1") == ["H1", "H2"]


def test_sample_candidate_records_only_mutates_requested_cdrs(monkeypatch):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    canonical_df = pd.DataFrame(
        [
            {
                "sequence_aa_heavy": make_heavy("AFTFSSYA", "AISGSGGSTYYA", "ARDRGGGMD"),
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "AFTFSSYA",
                "cdr2_aa_heavy": "AISGSGGSTYYA",
                "cdr3_aa_heavy": "ARDRGGGMD",
                "cdr1_aa_light": "QDISNYLA",
                "cdr2_aa_light": "YDASTRA",
                "cdr3_aa_light": "QQYNSYP",
                "redundancy": 10,
            },
            {
                "sequence_aa_heavy": make_heavy("GFTFSSYA", "SISGSGGSTYYA", "ARDRGGGMD"),
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "GFTFSSYA",
                "cdr2_aa_heavy": "SISGSGGSTYYA",
                "cdr3_aa_heavy": "ARDRGGGMD",
                "cdr1_aa_light": "QDISNYLA",
                "cdr2_aa_light": "YDASTRA",
                "cdr3_aa_light": "QQYNSYP",
                "redundancy": 8,
            },
            {
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": make_light("QDISNYLA", "YDASTRA", "LQYYNSY"),
                "cdr1_aa_heavy": "GFTFSSYA",
                "cdr2_aa_heavy": "AISGSGGSTYYA",
                "cdr3_aa_heavy": "ARDRGGGMD",
                "cdr1_aa_light": "QDISNYLA",
                "cdr2_aa_light": "YDASTRA",
                "cdr3_aa_light": "LQYYNSY",
                "redundancy": 50,
            },
        ]
    )

    candidates = sample_candidate_records(
        canonical_df=canonical_df,
        parental_heavy_sequence=PARENT_HEAVY,
        parental_light_sequence=PARENT_LIGHT,
        target_cdrs=["H1", "H2"],
        n_samples=3,
        joint_pattern_prob=0.0,
        single_substitution_mix_prob=1.0,
        per_position_edit_prob=0.5,
        seed=7,
    )

    assert len(candidates) == 3
    assert set(candidates["cdr3_aa_heavy"]) == {"ARDRGGGMD"}
    assert set(candidates["cdr1_aa_light"]) == {"QDISNYLA"}
    assert set(candidates["cdr2_aa_light"]) == {"YDASTRA"}
    assert set(candidates["cdr3_aa_light"]) == {"QQYNSYP"}
    observed_h1 = {"GFTFSSYA", "AFTFSSYA"}
    observed_h2 = {"AISGSGGSTYYA", "SISGSGGSTYYA"}
    assert set(candidates["cdr1_aa_heavy"]).issubset(observed_h1)
    assert set(candidates["cdr2_aa_heavy"]).issubset(observed_h2)
    for raw_mutated_cdrs in candidates["mutated_cdrs_actual"]:
        assert set(json.loads(raw_mutated_cdrs)).issubset({"H1", "H2"})


def test_sample_candidate_records_can_export_parental_when_requested(monkeypatch):
    monkeypatch.setattr("datasets.yeast_display.infer_antibody_regions", fake_regions)
    canonical_df = pd.DataFrame(
        [
            {
                "sequence_aa_heavy": PARENT_HEAVY,
                "sequence_aa_light": PARENT_LIGHT,
                "cdr1_aa_heavy": "GFTFSSYA",
                "cdr2_aa_heavy": "AISGSGGSTYYA",
                "cdr3_aa_heavy": "ARDRGGGMD",
                "cdr1_aa_light": "QDISNYLA",
                "cdr2_aa_light": "YDASTRA",
                "cdr3_aa_light": "QQYNSYP",
                "redundancy": 3,
            }
        ]
    )

    candidates = sample_candidate_records(
        canonical_df=canonical_df,
        parental_heavy_sequence=PARENT_HEAVY,
        parental_light_sequence=PARENT_LIGHT,
        target_cdrs=["H3"],
        n_samples=1,
        allow_parental=True,
        seed=1,
    )

    record = candidates.iloc[0].to_dict()
    assert record["sequence_aa_heavy"] == PARENT_HEAVY
    assert json.loads(record["mutations"]) == []
