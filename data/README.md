# Data Directory

**⚠️ IMPORTANT: Raw neuroimaging data should NEVER be committed to git!**

This directory contains documentation about data organization and structure.

## Directory Structure

```
data/
├── README.md               # This file
├── raw/                    # Raw/source data (gitignored)
├── processed/             # Processed feature matrices (gitignored)
└── interim/               # Intermediate outputs (gitignored)
```

## Data Sources

### Anatomical Data (CAT12)
- Gray matter volume (modulated)
- White matter volume (modulated)
- Cortical thickness

### Diffusion Data (QSIPrep/QSIRecon)
- DTI parameters (MD, FA, RD, AD)
- NODDI derivatives (ICVF, ODI, ISOVF)

### Questionnaire Data
- Demographics
- Mental health assessments
- Personality (Big 5)
- Lifestyle factors

## Usage

See the data loaders in `src/neuroalign/data/loaders/`
