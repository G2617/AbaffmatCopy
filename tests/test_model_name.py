# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

from utils.model_name import parse_protenix_model_name


def test_parse_standard_protenix_model_name():
    parsed = parse_protenix_model_name("protenix_base_default_v1.0.0")
    assert parsed == {
        "family": "protenix",
        "model_size": "base",
        "model_feature": "default",
        "model_version": "v1.0.0",
    }


def test_parse_scaled_up_protenix_model_name():
    parsed = parse_protenix_model_name("protenix-v2")
    assert parsed == {
        "family": "protenix",
        "model_size": "v2",
        "model_feature": "scaled-up",
        "model_version": "v2",
    }


def test_parse_unknown_model_name_returns_none():
    assert parse_protenix_model_name("custom-backbone") is None
