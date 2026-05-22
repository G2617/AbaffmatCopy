# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import pytest

torch = pytest.importorskip("torch")

from models.scoring_heads import (
    AffinityHead,
    BinderHead,
    GatedPoolingHead,
    PairformerRefinementHead,
    resolve_scoring_head_class,
)


def test_scoring_head_registry_supports_clear_names():
    assert resolve_scoring_head_class("gated_pooling") is GatedPoolingHead
    assert resolve_scoring_head_class("pairformer_refine") is PairformerRefinementHead


def test_scoring_head_registry_supports_legacy_aliases():
    assert resolve_scoring_head_class("affinity_head") is GatedPoolingHead
    assert resolve_scoring_head_class("binder_head") is PairformerRefinementHead


def test_scoring_head_alias_classes_point_to_existing_implementations():
    assert AffinityHead is GatedPoolingHead
    assert BinderHead is PairformerRefinementHead


def test_scoring_head_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unsupported scoring head"):
        resolve_scoring_head_class("mystery_head")
