# NeuroAlign 🧠

**Align brains through regional age patterns - Find your brain twins**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

**NeuroAlign** extends traditional brain age prediction to the **regional level**, creating rich embedding spaces that capture nuanced patterns of brain aging. By representing each participant as a vector of regional Brain Age Gap (BAG) values, we can:

- **Find similar brains** using nearest-neighbor retrieval
- **Interpret aging patterns** with LLM-powered analysis
- **Visualize brain similarity** through interactive 3D visualizations
- **Link brain structure to behavior** via questionnaire data integration

### The Pipeline

```
MRI Scans (Anatomical + Diffusion)
    ↓
Regional Feature Extraction (GM, WM, Microstructure)
    ↓
Multi-Modal Brain Age Prediction
    ↓
Regional BAG Computation (predicted - actual age per region)
    ↓
Embedding Space (participants as BAG vectors)
    ↓
Similarity Retrieval (find nearest neighbors)
    ↓
LLM Interpretation ("Brains like yours belong to people who...")
    ↓
Interactive Web Demo
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuroalign.git
cd neuroalign

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[all]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your paths
nano .env
```

### Run the Demo

```bash
# Train regional BAG models
python -m scripts.train_regional_bag

# Build FAISS retrieval index
python -m scripts.build_retrieval_index

# Launch web app
python -m app.main
```

---

## 📊 Dataset

- **N = ~3,500 participants**
- **Modalities**: Structural MRI (T1w), Diffusion MRI
- **Features**: 
  - Gray matter volume (parcellated)
  - Cortical thickness
  - White matter microstructure (MD, RD, FA, NODDI derivatives)
- **Atlas**: 4S456Parcels (456 brain regions)
- **Questionnaire data**: Integrated behavioral/cognitive assessments

---

## 🏗️ Project Structure

```
neuroalign/
├── src/neuroalign/              # Main package
│   ├── data/
│   │   ├── loaders/            # Modality-specific data loaders
│   │   └── preprocessing/      # Feature extraction & normalization
│   ├── modeling/               # BAG prediction models
│   ├── embedding/              # Embedding space operations
│   ├── retrieval/              # Nearest neighbor search
│   ├── visualization/          # Plotting utilities
│   ├── agent/                  # LLM interpretation
│   └── utils/                  # Helper functions
├── notebooks/                   # Analysis notebooks
├── app/                         # React web application
├── scripts/                     # CLI scripts
├── tests/                       # Unit tests
├── data/                        # Data documentation
├── models/                      # Saved models
└── docs/                        # Documentation
```

---

## 🔬 Methodology

### 1. Regional Feature Extraction
- Parcellate whole-brain images using 4S456Parcels atlas
- Extract regional summaries per modality
- Normalize features

### 2. Brain Age Prediction
- Train separate models per region and modality
- Compare Ridge, XGBoost, and MLP approaches
- Nested cross-validation for hyperparameter tuning

### 3. BAG Computation
```python
BAG[region, modality] = predicted_age[region, modality] - chronological_age
```

### 4. Similarity Retrieval
- Build FAISS index for fast nearest-neighbor search
- Retrieve top-k most similar participants

### 5. LLM Interpretation
- Extract questionnaire profiles of similar participants
- Generate natural language insights with Claude

---

## 🌐 Web Demo

The interactive web application allows users to:
1. Upload MRI data or select pre-computed examples
2. View regional BAG profile on a 3D brain
3. Explore the embedding space
4. Discover their "brain twins"
5. Read AI-generated interpretations

**Tech Stack**: React, FastAPI, Plotly, Claude API

---

## 🧪 Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
pytest

# Format code
black src/ tests/ scripts/

# Lint
ruff check src/

# Type check
mypy src/
```

---

## 📚 Citation

```bibtex
@software{neuroalign,
  author = {Your Name},
  title = {NeuroAlign: Regional Brain Age Alignment for Similarity Analysis},
  year = {2025},
  url = {https://github.com/yourusername/neuroalign}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Neuroimaging: CAT12, QSIPrep, FreeSurfer
- Parcellation: Custom `parcellate` package
- LLM: Anthropic Claude
- Retrieval: FAISS (Meta AI)

---

## 📧 Contact

**Your Name** - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/neuroalign](https://github.com/yourusername/neuroalign)
