# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
CANONICALIZATION_STATE_VERSION = 1
EXPECTED_COLUMNS = [
    "sort_population",
    "sequence_aa_heavy",
    "sequence_aa_light",
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
    "Redundancy",
]
CDR_COLUMNS = [
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
]
CANONICAL_COLUMNS = [
    "sample_id",
    "antigen_name",
    "antigen_sequence",
    "sequence_aa_heavy",
    "sequence_aa_light",
    "cdr1_aa_heavy",
    "cdr2_aa_heavy",
    "cdr3_aa_heavy",
    "cdr1_aa_light",
    "cdr2_aa_light",
    "cdr3_aa_light",
    "sort_population",
    "library_name",
    "phase",
    "mutated_cdrs",
    "sort_gate_descriptor",
    "experimental_round",
    "parental_id",
    "redundancy",
    "label_raw",
    "label_std",
    "split",
]
CHAIN_CDR_COLUMNS = {
    "heavy": ["cdr1_aa_heavy", "cdr2_aa_heavy", "cdr3_aa_heavy"],
    "light": ["cdr1_aa_light", "cdr2_aa_light", "cdr3_aa_light"],
}
CHAIN_CDR_SCHEMES = {
    "heavy": ["imgt", "imgt", "imgt"],
    "light": ["imgt", "kabat", "imgt"],
}


def has_only_canonical_amino_acids(sequence: str) -> bool:
    return bool(sequence) and set(sequence).issubset(VALID_AA)


def parse_sort_population(sort_population: str) -> tuple[str, list[str]]:
    tokens = [token for token in str(sort_population).split("_") if token]
    phase = tokens[0] if tokens else "unknown"
    mutated_cdrs = []
    for token in tokens[1:]:
        match = re.fullmatch(r"([hl])([123])", token.lower())
        if match:
            mutated_cdrs.append(f"{match.group(1).upper()}{match.group(2)}")
    return phase, mutated_cdrs


def load_library_name_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    if path.endswith(".json"):
        with open(path, "r") as handle:
            return json.load(handle)
    mapping_df = pd.read_csv(path)
    if not {"sort_population", "library_name"}.issubset(mapping_df.columns):
        raise ValueError(
            "library name mapping must contain columns: sort_population, library_name"
        )
    return dict(zip(mapping_df["sort_population"], mapping_df["library_name"]))


def infer_antibody_regions(sequence: str, scheme: str = "imgt") -> dict[str, str]:
    try:
        from abnumber import Chain
    except ImportError as exc:
        raise RuntimeError(
            "abnumber is required to infer antibody CDR/framework regions when they are "
            "not fully provided in the input CSV. Install 'abnumber' and its "
            "ANARCII numbering backend in your environment."
        ) from exc

    try:
        chain = Chain(sequence, scheme=scheme, use_anarcii=True)
    except ModuleNotFoundError as exc:
        if exc.name == "anarcii":
            raise RuntimeError(
                "abnumber is installed, but its ANARCII numbering backend is "
                "missing. Reinstall the project dependencies with "
                "`python -m pip install -e .` and rerun "
                "prepare_yeast_display_data.py."
            ) from exc
        raise
    return {
        "fr1": str(chain.fr1_seq),
        "cdr1": str(chain.cdr1_seq),
        "fr2": str(chain.fr2_seq),
        "cdr2": str(chain.cdr2_seq),
        "fr3": str(chain.fr3_seq),
        "cdr3": str(chain.cdr3_seq),
        "fr4": str(chain.fr4_seq),
    }


def build_regions_from_sequence(sequence: str, cdr1: str, cdr2: str, cdr3: str) -> dict[str, str]:
    cursor = 0
    frameworks = []
    for cdr_name, cdr_value in (("cdr1", cdr1), ("cdr2", cdr2), ("cdr3", cdr3)):
        if not cdr_value:
            raise ValueError(f"Unable to infer {cdr_name} from antibody sequence")
        match_index = sequence.find(cdr_value, cursor)
        if match_index < 0:
            raise ValueError(
                f"Unable to align {cdr_name} back to the antibody sequence under the configured numbering schemes"
            )
        frameworks.append(sequence[cursor:match_index])
        cursor = match_index + len(cdr_value)
    frameworks.append(sequence[cursor:])
    return {
        "fr1": frameworks[0],
        "cdr1": cdr1,
        "fr2": frameworks[1],
        "cdr2": cdr2,
        "fr3": frameworks[2],
        "cdr3": cdr3,
        "fr4": frameworks[3],
    }


def infer_chain_regions(sequence: str, chain_key: str) -> dict[str, str]:
    if chain_key == "heavy":
        return infer_antibody_regions(sequence, scheme="imgt")
    if chain_key != "light":
        raise ValueError(f"Unsupported chain key: {chain_key}")

    imgt_regions = infer_antibody_regions(sequence, scheme="imgt")
    kabat_regions = infer_antibody_regions(sequence, scheme="kabat")
    return build_regions_from_sequence(
        sequence=sequence,
        cdr1=imgt_regions["cdr1"],
        cdr2=kabat_regions["cdr2"],
        cdr3=imgt_regions["cdr3"],
    )


def get_framework_sequence(regions: dict[str, str]) -> str:
    return "".join(regions[key] for key in ("fr1", "fr2", "fr3", "fr4"))


def get_observed_cdr_values(row: pd.Series, chain_key: str) -> dict[str, str]:
    return {
        column: str(row[column]).strip().upper() if pd.notna(row[column]) else ""
        for column in CHAIN_CDR_COLUMNS[chain_key]
    }


def fill_and_validate_cdr_columns(
    row: pd.Series,
    chain_key: str,
    inferred_regions: dict[str, str],
) -> dict[str, str]:
    resolved = {}
    for index, (column, scheme) in enumerate(
        zip(CHAIN_CDR_COLUMNS[chain_key], CHAIN_CDR_SCHEMES[chain_key]), start=1
    ):
        inferred_value = inferred_regions[f"cdr{index}"]
        observed_value = str(row[column]).strip() if pd.notna(row[column]) else ""
        if observed_value:
            if inferred_value and observed_value != inferred_value:
                raise ValueError(
                    f"{column} value does not match {scheme.upper()}-numbered region for sample"
                )
            resolved[column] = observed_value
        else:
            if not inferred_value:
                raise ValueError(f"Unable to infer {column} from antibody sequence")
            resolved[column] = inferred_value
    return resolved


def resolve_chain_regions(
    row: pd.Series,
    chain_key: str,
    sequence: str,
) -> tuple[dict[str, str], dict[str, str]]:
    observed_cdrs = get_observed_cdr_values(row, chain_key)
    observed_values = list(observed_cdrs.values())
    if all(observed_values):
        return (
            build_regions_from_sequence(
                sequence=sequence,
                cdr1=observed_values[0],
                cdr2=observed_values[1],
                cdr3=observed_values[2],
            ),
            observed_cdrs,
        )

    inferred_regions = infer_chain_regions(sequence, chain_key)
    resolved_cdrs = fill_and_validate_cdr_columns(row, chain_key, inferred_regions)
    return (
        build_regions_from_sequence(
            sequence=sequence,
            cdr1=resolved_cdrs[CHAIN_CDR_COLUMNS[chain_key][0]],
            cdr2=resolved_cdrs[CHAIN_CDR_COLUMNS[chain_key][1]],
            cdr3=resolved_cdrs[CHAIN_CDR_COLUMNS[chain_key][2]],
        ),
        resolved_cdrs,
    )


def make_sample_id(record: dict[str, Any]) -> str:
    raw_key = "|".join(
        [
            record["antigen_name"],
            record["sort_population"],
            record["sequence_aa_heavy"],
            record["sequence_aa_light"],
        ]
    )
    digest = hashlib.sha1(raw_key.encode("utf8")).hexdigest()[:16]
    return f"{record['phase']}_{digest}"


def validate_split_fractions(val_fraction: float, test_fraction: float) -> None:
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be nonnegative")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")


def compute_split_counts(
    n_total: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    if n_total < 3:
        return 0, 0
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    if test_fraction > 0:
        n_test = max(1, n_test)
    if val_fraction > 0:
        n_val = max(1, n_val)
    if n_test + n_val >= n_total:
        n_test = 1 if test_fraction > 0 else 0
        n_val = 1 if val_fraction > 0 else 0
    return n_test, n_val


def assign_library_splits(
    library_sizes: dict[str, int],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    validate_split_fractions(val_fraction, test_fraction)
    unique_libraries = sorted(library_sizes)
    if not unique_libraries:
        return {}
    shuffled = list(unique_libraries)
    random_state = pd.Series(range(len(shuffled))).sample(
        frac=1.0, random_state=seed
    )
    shuffled = [shuffled[idx] for idx in random_state.tolist()]
    n_test, n_val = compute_split_counts(len(shuffled), val_fraction, test_fraction)
    split_map = {}
    for index, library_name in enumerate(shuffled):
        if index < n_test:
            split_map[library_name] = "test"
        elif index < n_test + n_val:
            split_map[library_name] = "val"
        else:
            split_map[library_name] = "train"
    return split_map


def stable_group_seed(seed: int, group_name: str) -> int:
    digest = hashlib.sha1(f"{seed}|{group_name}".encode("utf8")).hexdigest()[:8]
    return int(digest, 16)


def assign_sample_splits(
    curated_df: pd.DataFrame,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    stratify_column: str = "library_name",
) -> pd.Series:
    validate_split_fractions(val_fraction, test_fraction)
    split = pd.Series("train", index=curated_df.index, dtype="object")
    for group_name, group_df in curated_df.groupby(stratify_column, sort=True, dropna=False):
        shuffled_indices = (
            group_df.sample(
                frac=1.0,
                random_state=stable_group_seed(seed, str(group_name)),
            )
            .index
            .tolist()
        )
        n_test, n_val = compute_split_counts(len(shuffled_indices), val_fraction, test_fraction)
        split.loc[shuffled_indices[:n_test]] = "test"
        split.loc[shuffled_indices[n_test : n_test + n_val]] = "val"
    return split


def write_split_manifests(canonical_df: pd.DataFrame, canonical_output_path: str) -> dict[str, str]:
    output_dir = Path(canonical_output_path).resolve().parent
    stem = Path(canonical_output_path).stem
    manifest_paths = {}
    for split_name in ("train", "val", "test"):
        manifest_path = output_dir / f"{stem}_{split_name}_ids.txt"
        sample_ids = canonical_df.loc[canonical_df["split"] == split_name, "sample_id"]
        with open(manifest_path, "w") as handle:
            for sample_id in sample_ids.tolist():
                handle.write(f"{sample_id}\n")
        manifest_paths[split_name] = str(manifest_path)
    return manifest_paths


def write_json_samples(
    canonical_df: pd.DataFrame,
    json_output_dir: str,
    antigen_sequence: str,
) -> None:
    output_dir = Path(json_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_labels = {}
    for record in canonical_df.to_dict("records"):
        payload = [
            {
                "name": record["sample_id"],
                "sequences": [
                    {"proteinChain": {"sequence": antigen_sequence, "count": 1}},
                    {"proteinChain": {"sequence": record["sequence_aa_heavy"], "count": 1}},
                    {"proteinChain": {"sequence": record["sequence_aa_light"], "count": 1}},
                ],
            }
        ]
        with open(output_dir / f"{record['sample_id']}.json", "w") as handle:
            json.dump(payload, handle)
        id_labels[record["sample_id"]] = float(record["label_std"])

    with open(output_dir / "id_labels.pkl", "wb") as handle:
        import pickle

        pickle.dump(id_labels, handle)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_canonicalization_state(
    state_path: str,
    next_position: int,
    raw_num_rows: int,
    accepted_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> None:
    if not state_path:
        return
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    state = {
        "version": CANONICALIZATION_STATE_VERSION,
        "next_position": next_position,
        "raw_num_rows": raw_num_rows,
        "accepted_records": accepted_records,
        "rejected_records": rejected_records,
    }
    with open(tmp_path, "wb") as handle:
        pickle.dump(state, handle)
    os.replace(tmp_path, path)


def load_canonicalization_state(
    state_path: str,
    raw_num_rows: int,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    if not state_path or not os.path.exists(state_path):
        return 0, [], []
    with open(state_path, "rb") as handle:
        state = pickle.load(handle)
    if state.get("version") != CANONICALIZATION_STATE_VERSION:
        raise ValueError(
            f"Unsupported canonicalization state version in {state_path}: {state.get('version')}"
        )
    if state.get("raw_num_rows") != raw_num_rows:
        raise ValueError(
            "Canonicalization state does not match the current input row count. "
            "Remove the state file or use the same input CSV."
        )
    return (
        int(state.get("next_position", 0)),
        list(state.get("accepted_records", [])),
        list(state.get("rejected_records", [])),
    )


def canonicalize_yeast_display_dataframe(
    raw_df: pd.DataFrame,
    antigen_name: str,
    antigen_sequence: str,
    experimental_round: str,
    parental_id: str,
    parental_heavy_sequence: str,
    parental_light_sequence: str,
    library_name_map: dict[str, str] | None = None,
    split_seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_strategy: str = "sample",
    allow_offtarget_framework_mutations: bool = False,
    allow_offtarget_cdr_mutations: bool = False,
    checkpoint_interval: int = 0,
    checkpoint_state_path: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_strategy not in {"sample", "library"}:
        raise ValueError("split_strategy must be either 'sample' or 'library'")
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not has_only_canonical_amino_acids(parental_heavy_sequence):
        raise ValueError("parental heavy sequence contains noncanonical residues")
    if not has_only_canonical_amino_acids(parental_light_sequence):
        raise ValueError("parental light sequence contains noncanonical residues")

    library_name_map = library_name_map or {}
    region_cache: dict[tuple[str, str], dict[str, str]] = {}

    def cached_infer_chain_regions(sequence: str, chain_key: str) -> dict[str, str]:
        key = (chain_key, sequence)
        if key not in region_cache:
            region_cache[key] = infer_chain_regions(sequence, chain_key)
        return region_cache[key]

    parent_heavy_regions = cached_infer_chain_regions(parental_heavy_sequence, "heavy")
    parent_light_regions = cached_infer_chain_regions(parental_light_sequence, "light")
    parent_frameworks = {
        "heavy": get_framework_sequence(parent_heavy_regions),
        "light": get_framework_sequence(parent_light_regions),
    }
    parent_cdrs = {
        "H1": parent_heavy_regions["cdr1"],
        "H2": parent_heavy_regions["cdr2"],
        "H3": parent_heavy_regions["cdr3"],
        "L1": parent_light_regions["cdr1"],
        "L2": parent_light_regions["cdr2"],
        "L3": parent_light_regions["cdr3"],
    }

    raw_num_rows = len(raw_df)
    start_position, accepted_records, rejected_records = load_canonicalization_state(
        checkpoint_state_path,
        raw_num_rows,
    )
    row_iterator = tqdm(
        raw_df.iloc[start_position:].iterrows(),
        total=raw_num_rows,
        initial=start_position,
        desc="Canonicalizing yeast-display rows",
        unit="row",
    )
    for position, (row_index, raw_row) in enumerate(row_iterator, start=start_position):
        row = raw_row.copy()
        try:
            heavy_sequence = str(row["sequence_aa_heavy"]).strip().upper()
            light_sequence = str(row["sequence_aa_light"]).strip().upper()
            if not has_only_canonical_amino_acids(heavy_sequence):
                raise ValueError("heavy sequence contains noncanonical residues")
            if not has_only_canonical_amino_acids(light_sequence):
                raise ValueError("light sequence contains noncanonical residues")

            heavy_regions, heavy_cdrs = resolve_chain_regions(row, "heavy", heavy_sequence)
            light_regions, light_cdrs = resolve_chain_regions(row, "light", light_sequence)
            resolved_cdrs = {}
            resolved_cdrs.update(heavy_cdrs)
            resolved_cdrs.update(light_cdrs)

            heavy_framework = get_framework_sequence(heavy_regions)
            light_framework = get_framework_sequence(light_regions)
            if (
                not allow_offtarget_framework_mutations
                and heavy_framework != parent_frameworks["heavy"]
            ):
                raise ValueError("heavy framework deviates from parental reference")
            if (
                not allow_offtarget_framework_mutations
                and light_framework != parent_frameworks["light"]
            ):
                raise ValueError("light framework deviates from parental reference")

            phase, mutated_cdrs = parse_sort_population(row["sort_population"])
            if not allow_offtarget_cdr_mutations:
                for cdr_name, expected_value in parent_cdrs.items():
                    if cdr_name in mutated_cdrs:
                        continue
                    chain_name = "heavy" if cdr_name.startswith("H") else "light"
                    column_name = f"cdr{cdr_name[1]}_aa_{chain_name}"
                    if resolved_cdrs[column_name] != expected_value:
                        raise ValueError(f"{cdr_name} deviates from parental reference")

            redundancy = int(row["Redundancy"])
            if redundancy < 1:
                raise ValueError("redundancy must be >= 1")

            record = {
                "antigen_name": antigen_name,
                "antigen_sequence": antigen_sequence,
                "sequence_aa_heavy": heavy_sequence,
                "sequence_aa_light": light_sequence,
                "sort_population": str(row["sort_population"]).strip(),
                "library_name": library_name_map.get(
                    str(row["sort_population"]).strip(),
                    str(row["sort_population"]).strip(),
                ),
                "phase": phase,
                "mutated_cdrs": json.dumps(mutated_cdrs),
                "sort_gate_descriptor": str(row["sort_population"]).strip(),
                "experimental_round": experimental_round,
                "parental_id": parental_id,
                "redundancy": redundancy,
            }
            record.update(resolved_cdrs)
            accepted_records.append(record)
        except Exception as exc:
            rejected_record = raw_row.to_dict()
            rejected_record["reason"] = str(exc)
            rejected_record["row_index"] = row_index
            rejected_records.append(rejected_record)
        next_position = position + 1
        if checkpoint_interval > 0 and next_position % checkpoint_interval == 0:
            save_canonicalization_state(
                checkpoint_state_path,
                next_position,
                raw_num_rows,
                accepted_records,
                rejected_records,
            )

    save_canonicalization_state(
        checkpoint_state_path,
        raw_num_rows,
        raw_num_rows,
        accepted_records,
        rejected_records,
    )

    curated_df = pd.DataFrame(accepted_records)
    rejected_df = pd.DataFrame(rejected_records)
    if curated_df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS), rejected_df

    group_columns = [
        "antigen_name",
        "antigen_sequence",
        "sequence_aa_heavy",
        "sequence_aa_light",
        "cdr1_aa_heavy",
        "cdr2_aa_heavy",
        "cdr3_aa_heavy",
        "cdr1_aa_light",
        "cdr2_aa_light",
        "cdr3_aa_light",
        "sort_population",
        "library_name",
        "phase",
        "mutated_cdrs",
        "sort_gate_descriptor",
        "experimental_round",
        "parental_id",
    ]
    # Collapse duplicate sequence/metadata rows by summing their observed counts.
    curated_df = (
        curated_df.groupby(group_columns, dropna=False, as_index=False)["redundancy"]
        .sum()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    # Raw label is log10 abundance so multiplicative count changes become additive.
    curated_df["label_raw"] = curated_df["redundancy"].map(lambda value: math.log10(value))
    # Normalize within each library/round so labels are comparable only to peers from that run.
    group_key = ["library_name", "experimental_round"]
    group_mean = curated_df.groupby(group_key)["label_raw"].transform("mean")
    group_std = curated_df.groupby(group_key)["label_raw"].transform("std").fillna(0.0)
    # Standardized label is a z-score; constant groups are assigned 0 instead of dividing by 0.
    curated_df["label_std"] = (
        (curated_df["label_raw"] - group_mean)
        .where(group_std > 0, 0.0)
        .div(group_std.where(group_std > 0, 1.0))
    )
    curated_df["sample_id"] = curated_df.apply(
        lambda row: make_sample_id(row.to_dict()),
        axis=1,
    )
    if split_strategy == "library":
        library_sizes = curated_df["library_name"].value_counts().to_dict()
        split_map = assign_library_splits(
            library_sizes,
            split_seed,
            val_fraction,
            test_fraction,
        )
        curated_df["split"] = curated_df["library_name"].map(split_map).fillna("train")
    else:
        curated_df["split"] = assign_sample_splits(
            curated_df,
            split_seed,
            val_fraction,
            test_fraction,
        )
    curated_df = curated_df[CANONICAL_COLUMNS]
    return curated_df, rejected_df
