#!/usr/bin/env bash
# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

set -euo pipefail

RELEASE_BASE_URL="${ABAFFMAT_RELEASE_BASE_URL:-https://github.com/eamonbyrne/abaffmat/releases/download}"
RELEASE_TAG="${ABAFFMAT_RAW_DATA_TAG:-data-aintibody-challenge1-raw-v0.1}"
OUT_DIR="${1:-data/AIntibody}"

mkdir -p "$OUT_DIR"

download_file() {
  local filename="$1"
  local url="${RELEASE_BASE_URL}/${RELEASE_TAG}/${filename}"
  local out_path="${OUT_DIR}/${filename}"

  echo "Downloading ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --output "$out_path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$out_path"
  else
    echo "Error: install curl or wget to download release assets." >&2
    exit 1
  fi
}

sha256_file() {
  local path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    echo "Error: install shasum or sha256sum to verify checksums." >&2
    exit 1
  fi
}

verify_file() {
  local expected_sha="$1"
  local filename="$2"
  local path="${OUT_DIR}/${filename}"
  local actual_sha
  actual_sha="$(sha256_file "$path")"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "Checksum mismatch for ${filename}" >&2
    echo "expected: ${expected_sha}" >&2
    echo "actual:   ${actual_sha}" >&2
    exit 1
  fi
  echo "Verified ${filename}"
}

download_file "AIntibody_COMPETITION_1.csv"
download_file "AIntibody_COMPETITION_1.manifest.json"
download_file "SHA256SUMS.txt"

verify_file "b2e32f73bef6ece2c0b91ef6ee0cc0b566214b19706ee3b912216c9e4beadcaa" "AIntibody_COMPETITION_1.csv"
verify_file "9d7cc428d3248765bec18548ea027282f4599bacc4fac71dd5a2aad51fd91b00" "AIntibody_COMPETITION_1.manifest.json"

echo "Raw AIntibody Challenge 1 data are ready under ${OUT_DIR}"
