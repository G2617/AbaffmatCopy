# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

from pathlib import Path


def load_ordered_sample_ids(path: str) -> list[str]:
    if not path:
        return []
    ordered_ids = []
    seen = set()
    for raw_line in Path(path).read_text().splitlines():
        sample_id = raw_line.strip()
        if not sample_id or sample_id in seen:
            continue
        ordered_ids.append(sample_id)
        seen.add(sample_id)
    return ordered_ids


def load_sample_id_set(path: str) -> set[str]:
    return set(load_ordered_sample_ids(path))
