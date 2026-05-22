# Contributing

AbAffMat is intended to be an open workspace for reproducible antibody affinity-ranking experiments. Contributions are most useful when they make checkpoint comparisons easier to reproduce on the same data, splits, and benchmark protocol.

## Setup

Use Python 3.11 and install the package from the repository root:

```bash
conda create --name abaffmat python=3.11
conda activate abaffmat
python -m pip install -e .
```

If your CUDA/PyTorch environment needs a platform-specific wheel, install PyTorch first using the official instructions, then rerun `python -m pip install -e .`.

Before opening a pull request, run at least the focused smoke tests:

```bash
python -m pip install pytest
python -m pytest tests/test_dpo_loss.py tests/test_yeast_display.py tests/test_candidate_pool.py
```

For changes that touch benchmarking, training arguments, scoring heads, or candidate generation, also run the relevant tests in `tests/`.

## Reproducible Results

When comparing models, keep these fixed unless the change is explicitly about one of them:

- raw yeast-display CSV or documented data release
- antigen sequence
- preprocessing command and flags
- `split_seed` and split manifests
- holdout ID file
- benchmark command
- `ranking_label_column`
- checkpoint path or release tag

The most useful result reports include:

- `outputs/yeast_display_benchmark/benchmark_summary.csv`
- the exact `holdout_ids_path`
- checkpoint filename, release URL, or SHA256 checksum
- training `config.yaml`
- train/validation/test split ID files
- a short note describing what changed relative to the baseline

## Checkpoint Submissions

Do not commit checkpoint files to git. Share them as GitHub Release assets, Hugging Face model artifacts, Zenodo records, or another stable external artifact.

A checkpoint submission should include:

- `best.pt`
- `config.yaml`
- `train_dpo.log`
- `benchmark_summary.csv`
- train/validation/test split manifests
- `SHA256SUMS.txt`
- short model-card notes covering intended use, training data, benchmark results, and limitations

Use clear names such as:

```text
abaffmat-gated-pooling-pairwise-logistic-v0.1
abaffmat-pairformer-refine-pairwise-logistic-v0.1
```

Treat submitted checkpoints as research baselines for relative ranking, not calibrated affinity predictors.

## Data Handling

Do not commit large raw datasets or generated training artifacts. Keep full CSV releases, canonicalized CSVs, checkpoints, and benchmark outputs as external or release artifacts unless they are intentionally tiny test fixtures.

If you add or update data handling, document:

- source URL or release reference
- download date
- expected local path
- SHA256 checksum
- preprocessing command
- accepted/rejected row counts

Small mock data files for tests are welcome when they are minimal and do not duplicate large datasets.

## Pull Requests

Good pull requests are narrow and reproducible. Please include:

- what changed
- why it matters
- commands you ran
- any benchmark impact
- any known limitations

For model or training changes, include before/after benchmark results on the same holdout manifest whenever feasible.

## Contribution Ideas

High-value areas include:

- stronger yeast-display data curation and audit reporting
- alternative scoring heads
- preference-loss ablations
- better validation and held-out benchmark protocols
- candidate-pool generation strategies
- checkpoint packaging and model cards
- independent experimental feedback on top-ranked candidates
