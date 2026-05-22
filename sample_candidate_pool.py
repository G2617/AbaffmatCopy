# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

from datasets.yeast_display import infer_chain_regions, load_dataframe


CDR_ORDER = ("H1", "H2", "H3", "L1", "L2", "L3")
CDR_TO_COLUMN = {
    "H1": "cdr1_aa_heavy",
    "H2": "cdr2_aa_heavy",
    "H3": "cdr3_aa_heavy",
    "L1": "cdr1_aa_light",
    "L2": "cdr2_aa_light",
    "L3": "cdr3_aa_light",
}
CDR_TO_CHAIN = {
    "H1": ("heavy", "cdr1"),
    "H2": ("heavy", "cdr2"),
    "H3": ("heavy", "cdr3"),
    "L1": ("light", "cdr1"),
    "L2": ("light", "cdr2"),
    "L3": ("light", "cdr3"),
}


@dataclass(frozen=True)
class ParentContext:
    heavy_regions: dict[str, str]
    light_regions: dict[str, str]
    parent_cdrs: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble an AuraBind-style candidate pool by recombining frequently observed "
            "substitutions and mutational patterns from a curated yeast-display table."
        )
    )
    parser.add_argument("--canonical_input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--parental_heavy_sequence", required=True)
    parser.add_argument("--parental_light_sequence", required=True)
    parser.add_argument(
        "--target_cdrs",
        required=True,
        help="Comma-separated CDR list to vary, e.g. H1,H2 or L3.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of unique candidate antibodies to sample.",
    )
    parser.add_argument(
        "--weight_column",
        default="redundancy",
        help="Curated-table column used to weight observed patterns.",
    )
    parser.add_argument(
        "--joint_pattern_prob",
        type=float,
        default=0.7,
        help="Probability of starting from an observed joint CDR pattern.",
    )
    parser.add_argument(
        "--single_substitution_mix_prob",
        type=float,
        default=0.35,
        help="Probability of applying an extra per-position substitution recombination step.",
    )
    parser.add_argument(
        "--per_position_edit_prob",
        type=float,
        default=0.2,
        help="Edit probability for each target CDR position during substitution mixing.",
    )
    parser.add_argument(
        "--max_position_choices",
        type=int,
        default=8,
        help="Maximum number of high-frequency residues to keep per mutable position.",
    )
    parser.add_argument(
        "--allow_parental",
        action="store_true",
        help="Allow the exact parental antibody to be emitted as a candidate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json_output_dir",
        default="",
        help="Optional directory of predictor-ready JSON payloads for the sampled candidates.",
    )
    parser.add_argument(
        "--antigen_sequence",
        default="",
        help="Optional antigen sequence used when writing JSON payloads. If omitted, a unique sequence is inferred from the input table.",
    )
    parser.add_argument(
        "--antigen_name",
        default="",
        help="Optional antigen name override for the output table.",
    )
    return parser.parse_args()


def parse_target_cdrs(raw_value: str) -> list[str]:
    tokens = [token.strip().upper() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("target_cdrs must contain at least one CDR")
    invalid = [token for token in tokens if token not in CDR_ORDER]
    if invalid:
        raise ValueError(f"Unsupported target CDRs: {invalid}")
    deduped = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


def build_parent_context(parental_heavy_sequence: str, parental_light_sequence: str) -> ParentContext:
    heavy_regions = infer_chain_regions(parental_heavy_sequence, "heavy")
    light_regions = infer_chain_regions(parental_light_sequence, "light")
    parent_cdrs = {
        "H1": heavy_regions["cdr1"],
        "H2": heavy_regions["cdr2"],
        "H3": heavy_regions["cdr3"],
        "L1": light_regions["cdr1"],
        "L2": light_regions["cdr2"],
        "L3": light_regions["cdr3"],
    }
    return ParentContext(
        heavy_regions=heavy_regions,
        light_regions=light_regions,
        parent_cdrs=parent_cdrs,
    )


def row_cdr_map(row: pd.Series) -> dict[str, str]:
    return {
        cdr_name: str(row[column_name]).strip().upper()
        for cdr_name, column_name in CDR_TO_COLUMN.items()
    }


def get_actual_mutated_cdrs(row_cdrs: dict[str, str], parent_cdrs: dict[str, str]) -> list[str]:
    return [cdr_name for cdr_name in CDR_ORDER if row_cdrs[cdr_name] != parent_cdrs[cdr_name]]


def get_actual_mutations(row_cdrs: dict[str, str], parent_cdrs: dict[str, str]) -> list[dict[str, Any]]:
    mutations = []
    for cdr_name in CDR_ORDER:
        parent_cdr = parent_cdrs[cdr_name]
        candidate_cdr = row_cdrs[cdr_name]
        if len(parent_cdr) != len(candidate_cdr):
            raise ValueError(f"Observed {cdr_name} length deviates from the parental layout")
        for position, (parent_aa, candidate_aa) in enumerate(zip(parent_cdr, candidate_cdr), start=1):
            if parent_aa == candidate_aa:
                continue
            mutations.append(
                {
                    "cdr": cdr_name,
                    "position": position,
                    "parent": parent_aa,
                    "mutant": candidate_aa,
                }
            )
    return mutations


def weight_from_row(row: pd.Series, weight_column: str) -> float:
    if weight_column not in row.index:
        return 1.0
    value = row[weight_column]
    if pd.isna(value):
        return 1.0
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0:
        return 1.0
    return weight


def filter_support_rows(
    canonical_df: pd.DataFrame,
    target_cdrs: list[str],
    parent_cdrs: dict[str, str],
) -> pd.DataFrame:
    target_set = set(target_cdrs)
    support_mask = []
    for _, row in canonical_df.iterrows():
        row_cdrs = row_cdr_map(row)
        actual_mutated_cdrs = set(get_actual_mutated_cdrs(row_cdrs, parent_cdrs))
        support_mask.append(actual_mutated_cdrs.issubset(target_set))
    support_df = canonical_df.loc[support_mask].reset_index(drop=True)
    if support_df.empty:
        raise ValueError(
            "No support rows remain after restricting mutations to the requested target CDRs"
        )
    return support_df


def weighted_choice(
    rng: Random,
    counter: Counter[Any],
    fallback: Any,
) -> Any:
    if not counter:
        return fallback
    population = list(counter.keys())
    weights = list(counter.values())
    return rng.choices(population, weights=weights, k=1)[0]


def build_joint_pattern_key(target_cdrs: list[str], row_cdrs: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple((cdr_name, row_cdrs[cdr_name]) for cdr_name in target_cdrs)


def assemble_chain_sequence(chain_regions: dict[str, str], cdrs: dict[str, str]) -> str:
    return "".join(
        [
            chain_regions["fr1"],
            cdrs["cdr1"],
            chain_regions["fr2"],
            cdrs["cdr2"],
            chain_regions["fr3"],
            cdrs["cdr3"],
            chain_regions["fr4"],
        ]
    )


def build_model_statistics(
    support_df: pd.DataFrame,
    target_cdrs: list[str],
    parent_cdrs: dict[str, str],
    weight_column: str,
    max_position_choices: int,
) -> dict[str, Any]:
    joint_patterns: Counter[Any] = Counter()
    cdr_patterns: dict[str, Counter[str]] = {cdr_name: Counter() for cdr_name in target_cdrs}
    position_substitutions: dict[tuple[str, int], Counter[str]] = {}
    for cdr_name in target_cdrs:
        parent_cdr = parent_cdrs[cdr_name]
        for index in range(1, len(parent_cdr) + 1):
            position_substitutions[(cdr_name, index)] = Counter()

    for _, row in support_df.iterrows():
        row_cdrs = row_cdr_map(row)
        weight = weight_from_row(row, weight_column)
        joint_patterns[build_joint_pattern_key(target_cdrs, row_cdrs)] += weight
        for cdr_name in target_cdrs:
            cdr_patterns[cdr_name][row_cdrs[cdr_name]] += weight
            for index, (parent_aa, row_aa) in enumerate(
                zip(parent_cdrs[cdr_name], row_cdrs[cdr_name]),
                start=1,
            ):
                if row_aa != parent_aa:
                    position_substitutions[(cdr_name, index)][row_aa] += weight

    truncated_position_substitutions = {}
    for position_key, counter in position_substitutions.items():
        truncated = Counter(dict(counter.most_common(max_position_choices)))
        truncated_position_substitutions[position_key] = truncated

    return {
        "joint_patterns": joint_patterns,
        "cdr_patterns": cdr_patterns,
        "position_substitutions": truncated_position_substitutions,
    }


def sample_candidate_records(
    canonical_df: pd.DataFrame,
    parental_heavy_sequence: str,
    parental_light_sequence: str,
    target_cdrs: list[str],
    n_samples: int,
    weight_column: str = "redundancy",
    joint_pattern_prob: float = 0.7,
    single_substitution_mix_prob: float = 0.35,
    per_position_edit_prob: float = 0.2,
    max_position_choices: int = 8,
    allow_parental: bool = False,
    seed: int = 42,
    antigen_name: str = "",
) -> pd.DataFrame:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if not 0.0 <= joint_pattern_prob <= 1.0:
        raise ValueError("joint_pattern_prob must be between 0 and 1")
    if not 0.0 <= single_substitution_mix_prob <= 1.0:
        raise ValueError("single_substitution_mix_prob must be between 0 and 1")
    if not 0.0 <= per_position_edit_prob <= 1.0:
        raise ValueError("per_position_edit_prob must be between 0 and 1")

    parent = build_parent_context(parental_heavy_sequence, parental_light_sequence)
    support_df = filter_support_rows(canonical_df, target_cdrs, parent.parent_cdrs)
    statistics = build_model_statistics(
        support_df=support_df,
        target_cdrs=target_cdrs,
        parent_cdrs=parent.parent_cdrs,
        weight_column=weight_column,
        max_position_choices=max_position_choices,
    )

    rng = Random(seed)
    seen_pairs: set[tuple[str, str]] = set()
    sampled_records = []
    max_attempts = max(1000, n_samples * 100)

    parent_light_cdrs = {
        "cdr1": parent.light_regions["cdr1"],
        "cdr2": parent.light_regions["cdr2"],
        "cdr3": parent.light_regions["cdr3"],
    }
    parent_heavy_cdrs = {
        "cdr1": parent.heavy_regions["cdr1"],
        "cdr2": parent.heavy_regions["cdr2"],
        "cdr3": parent.heavy_regions["cdr3"],
    }

    for _ in range(max_attempts):
        if len(sampled_records) >= n_samples:
            break

        candidate_cdrs = dict(parent.parent_cdrs)
        if statistics["joint_patterns"] and rng.random() < joint_pattern_prob:
            base_mode = "joint_pattern"
            joint_pattern = weighted_choice(
                rng,
                statistics["joint_patterns"],
                build_joint_pattern_key(target_cdrs, candidate_cdrs),
            )
            for cdr_name, cdr_sequence in joint_pattern:
                candidate_cdrs[cdr_name] = cdr_sequence
        else:
            base_mode = "per_cdr_recombination"
            for cdr_name in target_cdrs:
                candidate_cdrs[cdr_name] = weighted_choice(
                    rng,
                    statistics["cdr_patterns"][cdr_name],
                    parent.parent_cdrs[cdr_name],
                )

        used_substitution_mix = False
        if rng.random() < single_substitution_mix_prob:
            for cdr_name in target_cdrs:
                mutable_counter = statistics["position_substitutions"]
                residues = list(candidate_cdrs[cdr_name])
                for index in range(1, len(residues) + 1):
                    residue_counter = mutable_counter[(cdr_name, index)]
                    if not residue_counter or rng.random() >= per_position_edit_prob:
                        continue
                    residues[index - 1] = weighted_choice(
                        rng,
                        residue_counter,
                        residues[index - 1],
                    )
                    used_substitution_mix = True
                candidate_cdrs[cdr_name] = "".join(residues)

        heavy_cdrs = dict(parent_heavy_cdrs)
        light_cdrs = dict(parent_light_cdrs)
        for cdr_name, (chain_name, region_name) in CDR_TO_CHAIN.items():
            if chain_name == "heavy":
                heavy_cdrs[region_name] = candidate_cdrs[cdr_name]
            else:
                light_cdrs[region_name] = candidate_cdrs[cdr_name]

        heavy_sequence = assemble_chain_sequence(parent.heavy_regions, heavy_cdrs)
        light_sequence = assemble_chain_sequence(parent.light_regions, light_cdrs)
        pair_key = (heavy_sequence, light_sequence)
        if not allow_parental and pair_key == (parental_heavy_sequence, parental_light_sequence):
            continue
        if pair_key in seen_pairs:
            continue

        actual_mutations = get_actual_mutations(candidate_cdrs, parent.parent_cdrs)
        actual_mutated_cdrs = sorted({mutation["cdr"] for mutation in actual_mutations})
        candidate_hash = hashlib.sha1(
            f"{heavy_sequence}|{light_sequence}".encode("utf8")
        ).hexdigest()[:12]
        record = {
            "candidate_id": f"cand_{len(sampled_records):05d}_{candidate_hash}",
            "antigen_name": antigen_name,
            "target_cdrs": json.dumps(target_cdrs),
            "source_mode": (
                f"{base_mode}+substitution_mix" if used_substitution_mix else base_mode
            ),
            "sequence_aa_heavy": heavy_sequence,
            "sequence_aa_light": light_sequence,
            "cdr1_aa_heavy": candidate_cdrs["H1"],
            "cdr2_aa_heavy": candidate_cdrs["H2"],
            "cdr3_aa_heavy": candidate_cdrs["H3"],
            "cdr1_aa_light": candidate_cdrs["L1"],
            "cdr2_aa_light": candidate_cdrs["L2"],
            "cdr3_aa_light": candidate_cdrs["L3"],
            "mutated_cdrs_actual": json.dumps(actual_mutated_cdrs),
            "mutation_count": len(actual_mutations),
            "mutations": json.dumps(actual_mutations),
        }
        sampled_records.append(record)
        seen_pairs.add(pair_key)

    if len(sampled_records) < n_samples:
        raise RuntimeError(
            f"Requested {n_samples} unique candidates but only sampled {len(sampled_records)} "
            "before exhausting the supported recombination space"
        )

    return pd.DataFrame(sampled_records)


def infer_antigen_sequence(canonical_df: pd.DataFrame, override: str) -> str:
    if override:
        return override
    if "antigen_sequence" not in canonical_df.columns:
        raise ValueError(
            "antigen_sequence was not provided and the curated table does not contain an antigen_sequence column"
        )
    unique_sequences = [str(value) for value in canonical_df["antigen_sequence"].dropna().unique()]
    if len(unique_sequences) != 1:
        raise ValueError(
            "JSON export requires a single antigen sequence in the curated table or an explicit --antigen_sequence override"
        )
    return unique_sequences[0]


def write_candidate_jsons(
    candidate_df: pd.DataFrame,
    antigen_sequence: str,
    json_output_dir: str,
) -> None:
    output_dir = Path(json_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in candidate_df.to_dict("records"):
        payload = [
            {
                "name": record["candidate_id"],
                "sequences": [
                    {"proteinChain": {"sequence": antigen_sequence, "count": 1}},
                    {"proteinChain": {"sequence": record["sequence_aa_heavy"], "count": 1}},
                    {"proteinChain": {"sequence": record["sequence_aa_light"], "count": 1}},
                ],
            }
        ]
        with open(output_dir / f"{record['candidate_id']}.json", "w") as handle:
            json.dump(payload, handle)


def main() -> None:
    args = parse_args()
    target_cdrs = parse_target_cdrs(args.target_cdrs)
    canonical_df = load_dataframe(args.canonical_input_path)
    candidate_df = sample_candidate_records(
        canonical_df=canonical_df,
        parental_heavy_sequence=args.parental_heavy_sequence,
        parental_light_sequence=args.parental_light_sequence,
        target_cdrs=target_cdrs,
        n_samples=args.n_samples,
        weight_column=args.weight_column,
        joint_pattern_prob=args.joint_pattern_prob,
        single_substitution_mix_prob=args.single_substitution_mix_prob,
        per_position_edit_prob=args.per_position_edit_prob,
        max_position_choices=args.max_position_choices,
        allow_parental=args.allow_parental,
        seed=args.seed,
        antigen_name=args.antigen_name,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        candidate_df.to_parquet(output_path, index=False)
    else:
        candidate_df.to_csv(output_path, index=False)

    if args.json_output_dir:
        antigen_sequence = infer_antigen_sequence(canonical_df, args.antigen_sequence)
        write_candidate_jsons(candidate_df, antigen_sequence, args.json_output_dir)


if __name__ == "__main__":
    main()
