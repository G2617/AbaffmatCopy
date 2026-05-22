#!/usr/bin/env bash
# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

set -euo pipefail

RELEASE_BASE_URL="${ABAFFMAT_RELEASE_BASE_URL:-https://github.com/eamonbyrne/abaffmat/releases/download}"
RELEASE_TAG="${ABAFFMAT_CANONICAL_DATA_TAG:-data-aintibody-challenge1-canonical-v0.1}"
OUT_DIR="${1:-data/AIntibody/COMP1_canonical}"

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

while read -r expected_sha filename; do
  [[ -z "${expected_sha}" || -z "${filename}" ]] && continue
  download_file "$filename"
  verify_file "$expected_sha" "$filename"
done <<'CHECKSUMS'
ad15a5998e78331831106d5cf3d50532b8052e7605f36db29bfabbc6d35eb129 yeast_display_COMP1_canonical.csv
1b1e903ecf76224510f4d2a2a1ef4caf8431cc0ff21862b92fff73081b9e43a9 yeast_display_COMP1_rejected.csv
35981d79f30e5b5bd82c472becd7ddf849eaf4baf62a9f72b35f8d5372d58cef yeast_display_COMP1_canonical_train_ids.txt
0197a50dd02915ef72f81593e4660d4a228f5869884d56b9583672dfcb3b7f44 yeast_display_COMP1_canonical_val_ids.txt
1421a38a4462f8cd6c7f9439c16130ef8e2bfb097537c294c6aef0ada178c583 yeast_display_COMP1_canonical_test_ids.txt
c7d93afaf69f7f7ae07d9ca07ff2f875af4a9803df81188ec0128340b9feabc3 preprocessing_command.sh
57fa3b8d06006db171e8eaa4429f29cf85dffc8e0ec15828ad81d645a4910d84 canonical_data_manifest.json
CHECKSUMS

download_file "SHA256SUMS.txt"

echo "Canonical AIntibody Challenge 1 data are ready under ${OUT_DIR}"
