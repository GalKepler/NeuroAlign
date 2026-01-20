# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroAlign is a neuroimaging analysis platform that extends brain age prediction to the regional level. It creates embedding spaces from regional Brain Age Gap (BAG) values for similarity retrieval and LLM-powered interpretation.

**Pipeline**: MRI Data → Regional Feature Extraction → Brain Age Prediction → BAG Computation → Embedding Space → Similarity Retrieval → LLM Interpretation

## Common Commands

```bash
# Installation
uv sync --all-extras

# Run tests
pytest

# Code quality
black src/ tests/ scripts/
ruff check src/
mypy src/

# Data preparation CLI
uv run python -m neuroalign.data.preprocessing.cli --help

# Run scripts
python -m scripts.train_regional_bag
python -m scripts.build_retrieval_index
python -m app.main
```

## Architecture

### Core Data Flow

```
src/neuroalign/
├── data/
│   ├── loaders/              # Modality-specific data loading
│   │   ├── anatomical.py     # CAT12 loader (GM, WM, CT) with TIV calculation
│   │   ├── diffusion.py      # QSIPrep/QSIRecon loader (DTI, NODDI)
│   │   └── questionnaire.py  # Behavioral/cognitive assessments
│   └── preprocessing/
│       ├── pipeline.py       # DataPreparationPipeline orchestrator
│       ├── feature_store.py  # Two-tier storage (long/wide formats)
│       ├── transformers.py   # Long-to-wide format converters
│       └── config.py         # Pydantic configuration models
├── modeling/                 # Regional BAG prediction models (stub)
├── embedding/                # Embedding space operations (stub)
├── retrieval/                # FAISS-based similarity search (stub)
├── visualization/            # Plotting utilities (stub)
└── agent/                    # LLM interpretation (stub)
```

### Feature Store

The feature store uses two formats:
- **Long format**: Raw parcellator output preserving all columns (for exploration)
- **Wide format**: One parquet file per metric with regions as columns (for modeling)

```
data/processed/
├── long/                     # Raw parcellator output
│   ├── anatomical_gm.parquet
│   └── diffusion/AMICONODDI.parquet
├── wide/                     # Modeling-ready matrices
│   ├── anatomical/gm_volume_mm3.parquet
│   └── diffusion/...
├── tiv.parquet
├── metadata.parquet
└── manifest.json
```

### Key Patterns

- **Multiprocessing**: Data loaders use ProcessPoolExecutor with worker initialization for parallel loading
- **TIV Calculation**: Optional MATLAB/CAT12 integration via subprocess
- **Incremental Loading**: Pipeline tracks existing sessions to avoid reprocessing
- **Pydantic Config**: All configuration uses Pydantic models for validation

## Environment Configuration

Copy `.env.example` to `.env` and configure paths. Key variables:
- `CAT12_ROOT`, `QSIPARC_PATH`, `QSIRECON_PATH` - Data locations
- `ATLAS_NAME=4S456Parcels` - 456-region brain atlas
- `MATLAB_BIN`, `SPM_PATH`, `CAT12_PATH` - For TIV calculation (optional)
- `ANTHROPIC_API_KEY` - For LLM interpretation

## Technical Details

- **Atlas**: 4S456Parcels (456 brain regions) in MNI152NLin2009cAsym space
- **Parcellation**: Uses custom `parcellate` package (VolumetricParcellator)
- **Dependencies**: Requires nibabel, nilearn, faiss-cpu; optional MATLAB for TIV
- **Data Format**: Sessions identified by `subject_code` + `session_id` columns in CSV
