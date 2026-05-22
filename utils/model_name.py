# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

from __future__ import annotations


def parse_protenix_model_name(model_name: str) -> dict[str, str] | None:
    """Best-effort parser for upstream and locally adapted Protenix model names."""
    if model_name == "protenix-v2":
        return {
            "family": "protenix",
            "model_size": "v2",
            "model_feature": "scaled-up",
            "model_version": "v2",
        }

    parts = model_name.split("_")
    if len(parts) < 4 or parts[0] != "protenix":
        return None

    return {
        "family": parts[0],
        "model_size": parts[1],
        "model_feature": "_".join(parts[2:-1]),
        "model_version": parts[-1],
    }
