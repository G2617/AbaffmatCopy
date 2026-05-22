# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import argparse
from pathlib import Path

import pandas as pd

from datasets.yeast_display import (
    canonicalize_yeast_display_dataframe,
    load_library_name_map,
    save_dataframe,
    write_json_samples,
    write_split_manifests,
)


PARENTAL_SORT_POPULATION = "parental"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_yeast_csv_path", required=True)
    parser.add_argument("--canonical_output_path", required=True)
    parser.add_argument("--json_output_dir", required=True)
    parser.add_argument("--antigen_name", default="covid19_rbd")
    parser.add_argument("--antigen_sequence", required=True)
    parser.add_argument("--experimental_round", required=True)
    parser.add_argument("--parental_id", default=PARENTAL_SORT_POPULATION)
    parser.add_argument("--parental_heavy_sequence", default="")
    parser.add_argument("--parental_light_sequence", default="")
    parser.add_argument("--library_name_map_path", default="")
    parser.add_argument("--audit_output_path", default="")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument(
        "--split_strategy",
        choices=["sample", "library"],
        default="sample",
        help=(
            "sample assigns train/val/test rows within each library; library holds out "
            "entire libraries but can be highly imbalanced when library sizes differ"
        ),
    )
    parser.add_argument(
        "--allow_offtarget_framework_mutations",
        action="store_true",
        help="Accept rows whose heavy or light framework differs from the parental framework.",
    )
    parser.add_argument(
        "--allow_offtarget_cdr_mutations",
        action="store_true",
        help=(
            "Accept rows whose CDR mutations include CDRs not listed in sort_population. "
            "The original sort_population-derived mutated_cdrs value is still preserved."
        ),
    )
    parser.add_argument("--canonicalization_checkpoint_interval", type=int, default=0)
    parser.add_argument("--canonicalization_state_path", default="")
    return parser.parse_args()


def get_parental_row_mask(raw_df: pd.DataFrame) -> pd.Series:
    if "sort_population" not in raw_df.columns:
        raise ValueError("Missing required column: sort_population")
    return (
        raw_df["sort_population"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(PARENTAL_SORT_POPULATION)
    )


def infer_parental_sequences(raw_df: pd.DataFrame) -> tuple[str, str]:
    required_columns = ["sequence_aa_heavy", "sequence_aa_light"]
    missing_columns = [column for column in required_columns if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    parental_df = raw_df.loc[get_parental_row_mask(raw_df), required_columns].copy()
    if parental_df.empty:
        raise ValueError(
            "Could not infer parental sequences: no row has sort_population='parental'. "
            "Provide --parental_heavy_sequence and --parental_light_sequence to override."
        )

    parental_df["sequence_aa_heavy"] = (
        parental_df["sequence_aa_heavy"].fillna("").astype(str).str.strip().str.upper()
    )
    parental_df["sequence_aa_light"] = (
        parental_df["sequence_aa_light"].fillna("").astype(str).str.strip().str.upper()
    )
    parental_pairs = parental_df.drop_duplicates()
    invalid_pairs = parental_pairs[
        (parental_pairs["sequence_aa_heavy"] == "")
        | (parental_pairs["sequence_aa_light"] == "")
    ]
    if not invalid_pairs.empty:
        raise ValueError("Could not infer parental sequences: parental row has an empty sequence")
    if len(parental_pairs) != 1:
        raise ValueError(
            "Could not infer parental sequences: found multiple distinct parental heavy/light pairs"
        )

    parental_row = parental_pairs.iloc[0]
    return parental_row["sequence_aa_heavy"], parental_row["sequence_aa_light"]


def resolve_parental_sequences(args: argparse.Namespace, raw_df: pd.DataFrame) -> tuple[str, str]:
    parental_heavy_sequence = args.parental_heavy_sequence.strip().upper()
    parental_light_sequence = args.parental_light_sequence.strip().upper()
    has_heavy = bool(parental_heavy_sequence)
    has_light = bool(parental_light_sequence)
    if has_heavy != has_light:
        raise ValueError(
            "Provide both --parental_heavy_sequence and --parental_light_sequence, or omit both"
        )
    if has_heavy and has_light:
        return parental_heavy_sequence, parental_light_sequence
    return infer_parental_sequences(raw_df)


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(args.raw_yeast_csv_path, low_memory=False)
    parental_heavy_sequence, parental_light_sequence = resolve_parental_sequences(args, raw_df)
    raw_df = raw_df.loc[~get_parental_row_mask(raw_df)].copy()
    library_name_map = load_library_name_map(args.library_name_map_path)
    checkpoint_state_path = args.canonicalization_state_path
    if args.canonicalization_checkpoint_interval > 0 and not checkpoint_state_path:
        checkpoint_state_path = str(
            Path(args.canonical_output_path).with_suffix(".canonicalization_state.pkl")
        )
    canonical_df, rejected_df = canonicalize_yeast_display_dataframe(
        raw_df=raw_df,
        antigen_name=args.antigen_name,
        antigen_sequence=args.antigen_sequence,
        experimental_round=args.experimental_round,
        parental_id=args.parental_id,
        parental_heavy_sequence=parental_heavy_sequence,
        parental_light_sequence=parental_light_sequence,
        library_name_map=library_name_map,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_strategy=args.split_strategy,
        allow_offtarget_framework_mutations=args.allow_offtarget_framework_mutations,
        allow_offtarget_cdr_mutations=args.allow_offtarget_cdr_mutations,
        checkpoint_interval=args.canonicalization_checkpoint_interval,
        checkpoint_state_path=checkpoint_state_path,
    )
    output_path = Path(args.canonical_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(canonical_df, str(output_path))
    write_json_samples(canonical_df, args.json_output_dir, args.antigen_sequence)
    write_split_manifests(canonical_df, str(output_path))

    if args.audit_output_path:
        audit_path = Path(args.audit_output_path)
    else:
        audit_path = output_path.with_name(f"{output_path.stem}_rejected.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_df.to_csv(audit_path, index=False)


if __name__ == "__main__":
    main()
