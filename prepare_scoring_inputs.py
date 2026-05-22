# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_RBD_SEQUENCE = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNK"
)

HEAVY_COLUMN_CANDIDATES = [
    "sequence_aa_heavy",
    "heavy chain",
]
LIGHT_COLUMN_CANDIDATES = [
    "sequence_aa_light",
    "light chain",
]
NAME_COLUMN_CANDIDATES = [
    "candidate_id",
    "sample_id",
    "name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a candidate antibody CSV into JSON inputs for complex scoring."
    )
    parser.add_argument(
        "--input_csv_path",
        default="./data/Covid/Covid_design.csv",
        help="Path to the input CSV containing heavy/light antibody sequences.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory to write JSON payloads into. Defaults to <input_stem>/ next to the CSV.",
    )
    parser.add_argument(
        "--antigen_sequence",
        default=DEFAULT_RBD_SEQUENCE,
        help="Antigen sequence to prepend as the first protein chain in each JSON payload.",
    )
    parser.add_argument(
        "--heavy_column",
        default="",
        help="Optional override for the heavy-chain sequence column name.",
    )
    parser.add_argument(
        "--light_column",
        default="",
        help="Optional override for the light-chain sequence column name.",
    )
    parser.add_argument(
        "--name_column",
        default="",
        help="Optional override for the sample name column. Defaults to candidate_id/sample_id/name if present.",
    )
    parser.add_argument(
        "--name_prefix",
        default="idx",
        help="Prefix used for sample names when no name column is present.",
    )
    return parser.parse_args()


def resolve_output_dir(input_csv_path: str, output_dir: str) -> Path:
    if output_dir:
        return Path(output_dir)
    input_path = Path(input_csv_path)
    return input_path.with_suffix("")


def detect_column(df: pd.DataFrame, override: str, candidates: list[str], label: str) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"{label} column override not found: {override}")
        return override
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Unable to detect the {label} column. Tried: {', '.join(candidates)}"
    )


def detect_optional_column(df: pd.DataFrame, override: str, candidates: list[str]) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"name column override not found: {override}")
        return override
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return ""


def build_sample_name(row: pd.Series, row_index: int, name_column: str, name_prefix: str) -> str:
    if name_column:
        return str(row[name_column]).strip()
    return f"{name_prefix}_{row_index}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv_path)
    output_dir = resolve_output_dir(args.input_csv_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    heavy_column = detect_column(df, args.heavy_column, HEAVY_COLUMN_CANDIDATES, "heavy")
    light_column = detect_column(df, args.light_column, LIGHT_COLUMN_CANDIDATES, "light")
    name_column = detect_optional_column(df, args.name_column, NAME_COLUMN_CANDIDATES)

    for row_index, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        sample_name = build_sample_name(row, row_index, name_column, args.name_prefix)
        heavy_sequence = str(row[heavy_column]).strip()
        light_sequence = str(row[light_column]).strip()
        sequences = [
            {"proteinChain": {"sequence": args.antigen_sequence, "count": 1}},
            {"proteinChain": {"sequence": heavy_sequence, "count": 1}},
            {"proteinChain": {"sequence": light_sequence, "count": 1}},
        ]
        payload = [{"sequences": sequences, "name": sample_name}]
        with open(output_dir / f"{sample_name}.json", "w") as handle:
            json.dump(payload, handle)


if __name__ == "__main__":
    main()
