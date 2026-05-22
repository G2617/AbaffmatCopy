# Derived from Aureka-AIntibody-Challenges (MIT License; Copyright 2025 Jixian Zhang).
# Modifications Copyright 2026 Eamon Byrne.
# AbAffMat modifications are licensed under Apache-2.0; see LICENSE and THIRD_PARTY_NOTICES.md.

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="abaffmat",
    version="0.0.1",
    author="Eamon Byrne",
    author_email="eamon@eamonbyrneconsulting.com",
    description="A tool for in silico antibody affinity maturation and protein binding analysis",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/eamonbyrne/abaffmat.git",
    packages=find_packages(
        exclude=(
            "data",
            "data.*",
            "esm_embeddings",
            "esm_embeddings.*",
            "outputs",
            "outputs.*",
            "release_data",
            "release_data.*",
        )
    ),
    py_modules=[
        "benchmark_yeast_display",
        "data_preprocess",
        "prepare_scoring_inputs",
        "prepare_yeast_display_data",
        "sample_candidate_pool",
    ],
    package_data={
        "models.tri_attention": ["TriAttentionCache/**/*.json"],
    },
    install_requires=[
        "abnumber>=0.4.4",
        "anarci",
        "biotite",
        "fair-esm",
        "ml-collections",
        "numpy>=1.21",
        "optree",
        "pandas>=1.5",
        "pyyaml",
        "rdkit",
        "scikit-learn",
        "scipy",
        "torch>=2.0",
        "tqdm",
        "wandb",
    ],
    python_requires=">=3.11",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
