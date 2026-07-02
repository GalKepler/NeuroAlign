# Data Pipeline Migration Plan

## Why

The data landscape changed underneath NeuroAlign:

1. **Behavioral/demographic data** is now served by `brainlink` (`/home/galkepler/Projects/brainlink`), a SQLite DB + `BrainLinkDB` query API. This replaces the old `SESSIONS_CSV` / `QUESTIONNAIRE_CSV` flat files and `QuestionnaireLoader`.
2. **Neuroimaging-derived regional data** (anatomical + diffusion) is now pre-parcellated and stored as flat CSV/TSV tables under `/mnt/62/Processed_Data/derivatives/tabular/sub-<UID>/...`. This replaces the old `AnatomicalLoader` (CAT12 + MATLAB TIV) and `DiffusionLoader` (QSIPrep/QSIRecon on-the-fly parcellation).
3. **Regional BAG modeling** should use `regional-stacker` (`/home/galkepler/Projects/regional-stacker`)'s `RegionalStackingRegressor` for leakage-free, per-region nested-CV stacking with a meta-learner — replacing/extending the `modeling/univariate` stub.

This plan re-points the data preparation entry point at these three components and reshapes the feature store accordingly.

---

## 1. New Data Source Shapes

### 1.1 Behavioral data (brainlink)

```python
from brainlink import BrainLinkDB

db = BrainLinkDB("/home/galkepler/Projects/brainlink/brainlink.db")
df = db.query(
    include=["demographics", "questionnaires"],
    require_complete_mapping=True,
    styled=False,
)
```

Key columns: `session_id` (BIDS `YYYYMMddhhmm`), `uid` (BIDS participant id, e.g. `S653024` — matches `sub-<UID>` dirs in tabular derivatives), `subject_code`, `age_at_scan`, `sex`, plus pivoted questionnaire columns.

This becomes the **canonical session list + age/sex/covariates source**, replacing `sessions_csv` entirely.

### 1.2 Neuroimaging tabular derivatives

Root: `/mnt/62/Processed_Data/derivatives/tabular/sub-<UID>/...`

```
sub-<UID>/
├── anat/atlas-<Atlas>/sub-<UID>_atlas-<Atlas>_structure-{cortex,subcortex}.{csv,json}   # subject-level (template)
├── ses-<SESSIONID>/anat/atlas-<Atlas>/sub-<UID>_ses-<SESSIONID>_atlas-<Atlas>_structure-{cortex,subcortex}.{csv,json}
├── ses-<SESSIONID>.cross/anat/atlas-<Atlas>/...                                          # cross-sectional variant
└── ses-<SESSIONID>/dwi/atlas-<Atlas>/sub-<UID>_ses-<SESSIONID>_atlas-<Atlas>_software-<SW>_dir-AP_run-01_space-ACPC_model-<model>_param-<param>_diffmap.{tsv,json}
```

**Anatomical CSV columns** (per hemisphere x region row):
`subject_id, index, label, hemisphere, num_vertices, surface_area_mm2, gray_matter_volume_mm3, thickness_mean_mm, thickness_std_mm, mean_curvature, gaussian_curvature, folding_index, curvature_index, white_surf_area_mm2, brain_seg_vol_mm3, brain_seg_no_vent_mm3, cortex_vol_mm3, supratentorial_vol_mm3, tiv_mm3`

→ **TIV is already included per row** (`tiv_mm3`). The old MATLAB/CAT12 TIV subprocess step is no longer needed.

**Diffusion TSV columns** (per region row):
`index, label, hemisphere, volume_mm3, voxel_count, z_filtered_mean, z_filtered_std, iqr_filtered_mean, iqr_filtered_std, robust_mean, robust_std, mad_median, mean, std, median, sum, cv, robust_cv, skewness, excess_kurtosis, percentile_5, percentile_25, percentile_75, percentile_95, coverage, scalar`

`software`/`model`/`param`/`atlas` come from the **filename** (BIDS entities), not the table — reuse `parse_bids_entities()` from old `diffusion.py`.

**Atlases observed**: `Schaefer2018N400n7`, `Tian2020S2`, `Brainnetome246Ext`, `Gordon333Ext`, `HCPex`, `4S456Parcels`, `Schaefer2018N400n7Tian2020S2` (anat+subcortex combined, used for dwi). Connectome matrices (`*_connmatrix.{tsv,json}`) also exist per atlas — **confirmed out of scope** for this restart; revisit for future `embedding`/connectivity work.

**Default atlas: `Schaefer2018N400n7Tian2020S2`** (matches dwi naming directly). For anatomical data this atlas is split across two folders — `atlas-Schaefer2018N400n7` (cortex) + `atlas-Tian2020S2` (subcortex, *not* S3 despite `.env`'s stale `Schaefer2018N400n7Tian2020S3`) — the loader must load both and concatenate into one `Schaefer2018N400n7Tian2020S2` long-format table.

### 1.3 regional-stacker

```python
from regional_stacker import RegionalStackingRegressor, wide_to_stacker_input

X, region_mapping, subjects = wide_to_stacker_input({
    "ct_thickness_mean_mm": ct_wide_df,      # subjects x regions
    "dwi_DSIStudio_tensor_fa_mean": fa_wide_df,
    ...
})
stacker = RegionalStackingRegressor(region_mapping=region_mapping, ...)
stacker.fit(X_train, y_age_train)
bag = y_test - stacker.predict(X_test)
```

Requires subjects x regions wide tables with **aligned region names across modalities**. **Confirmed**: region labels already align across the anat (Schaefer2018N400n7+Tian2020S2) and dwi (Schaefer2018N400n7Tian2020S2) atlases — no harmonization mapping needed.

---

## 2. Architecture Changes

```
src/neuroalign/data/
├── loaders/
│   ├── behavioral.py          # NEW — thin wrapper around BrainLinkDB
│   ├── tabular_derivatives.py # NEW — replaces anatomical.py + diffusion.py
│   ├── anatomical.py          # REMOVE (CAT12/MATLAB-based, obsolete)
│   ├── anatomical_example.py  # REMOVE
│   ├── diffusion.py           # REMOVE (QSIPrep on-the-fly parcellation, obsolete)
│   └── questionnaire.py       # REMOVE (superseded by brainlink)
└── preprocessing/
    ├── config.py              # EDIT — new PipelineConfig (brainlink_db, tabular_root, atlases)
    ├── pipeline.py            # EDIT — new orchestration using behavioral.py + tabular_derivatives.py
    ├── feature_store.py        # EDIT — adjust metric lists + wide pivot for new schemas
    └── transformers.py         # EDIT if region-name harmonization needed

src/neuroalign/modeling/
├── config.py / result.py      # EXTEND for multivariate (per univariate_bag_plan.md)
├── univariate/                 # existing per-region univariate BAG (keep)
└── multivariate/               # NEW — regional-stacker-based multimodal regional BAG
    ├── __init__.py
    └── estimator.py            # wraps RegionalStackingRegressor + wide_to_stacker_input
```

### 2.1 `loaders/behavioral.py` (new)

```python
class BehavioralLoader:
    def __init__(self, db_path: Path): ...
    def get_sessions(self, labs=None, require_imaging=None) -> pd.DataFrame:
        """subject_code, uid, session_id, age_at_scan, sex, ..."""
    def get_questionnaires(self, instruments=None) -> pd.DataFrame: ...
```
Wraps `BrainLinkDB.query(...)`. Output columns renamed to pipeline conventions: `uid` ↔ subject dir id, `subject_code` kept for compatibility, `age_at_scan` → `AGE`.

### 2.2 `loaders/tabular_derivatives.py` (new)

```python
class TabularDerivativesLoader:
    def __init__(
        self,
        root: Path,
        atlas_name: str = "Schaefer2018N400n7Tian2020S2",
        anat_atlases: tuple[str, str] = ("Schaefer2018N400n7", "Tian2020S2"),
        session_variant: Literal["cross", "plain", "subject"] = "cross",
    ): ...
    def load_anatomical(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Long format: uid, session_id, atlas, structure, label, hemisphere, <metric columns>, tiv_mm3
        Loads anat_atlases (cortex + subcortex) per session and concatenates into one
        Schaefer2018N400n7Tian2020S2-labeled table.
        """
    def load_diffusion(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Long format: uid, session_id, atlas, software, model, param, label, hemisphere, <metric columns>"""
```
- For each `(uid, session_id)` from `sessions` df, resolve `sub-<uid>/ses-<session_id>{.cross,}/{anat,dwi}` per `session_variant`:
  - `"cross"` (**default**) → prefer `ses-<id>.cross/`, since this maximizes session coverage (every scan gets a cross-sectional reconstruction; longitudinal `ses-<id>/` may be missing for single-timepoint subjects).
  - `"plain"` → prefer `ses-<id>/` (longitudinal/base), falling back to `.cross` if absent.
  - `"subject"` → use subject-level `sub-<uid>/anat/` (no session breakdown; dwi has no subject-level equivalent, so this only affects anat).
  - Configurable via `PipelineConfig.session_variant`.
- Reads CSV/TSV directly — no parcellation, no multiprocessing dependency on `parcellate`/MATLAB (can keep light `ProcessPoolExecutor` purely for I/O parallelism over many sessions).
- Reuses `parse_bids_entities()` for dwi filenames.

### 2.3 `feature_store.py` (edit)

- `ANATOMICAL_METRICS` → new column set (`surface_area_mm2`, `gray_matter_volume_mm3`, `thickness_mean_mm`, `thickness_std_mm`, ... `tiv_mm3`).
- `DIFFUSION_METRICS` mostly unchanged but generated per `(atlas, software, model, param, metric)` instead of `(workflow, model, param, metric)`.
- `save_tiv` simplifies: extract `tiv_mm3` directly from anatomical long format (already per-row), no MATLAB step.
- Wide-format pivot keys on `label` (region) — verify region naming is consistent enough across atlases for `regional-stacker`'s alignment requirement, or add a region-name harmonization map in `transformers.py`.

### 2.4 `preprocessing/pipeline.py` (edit)

New `run()` flow:
1. `BehavioralLoader.get_sessions()` → canonical session list + AGE/sex (replaces `_load_sessions_csv`).
2. `TabularDerivativesLoader.load_anatomical/load_diffusion()` → long format per atlas/modality.
3. `FeatureStore` saves long + generates wide as before.
4. `store.save_metadata()` now sourced from brainlink (AGE, sex, etc.) instead of CSV merge.
5. Incremental loading (`get_existing_sessions`) logic unchanged.

### 2.5 `modeling/multivariate/estimator.py` (new)

```python
class MultivariateRegionalBAGEstimator:
    def __init__(self, config: BAGConfig, modalities: list[str]): ...
    def fit(self, store: FeatureStore, sessions: pd.DataFrame) -> BAGResult:
        tables = {feat: store.load_feature(feat, include_metadata=False) for feat in self.modalities}
        X, region_mapping, subjects = wide_to_stacker_input(tables)
        stacker = RegionalStackingRegressor(region_mapping=region_mapping, ...)
        ...
```
Reuses `BAGConfig`/`BAGResult` from `univariate_bag_plan.md` so univariate and multivariate share infra.

---

## 3. Dependency Changes (`pyproject.toml`)

Add:
```toml
dependencies = [
    ...
    "brainlink @ file:///home/galkepler/Projects/brainlink",
    "regional-stacker @ file:///home/galkepler/Projects/regional-stacker",
]
```

Remove (no longer needed once old loaders are deleted):
- `parcellate` (only needed for on-the-fly parcellation)
- `gam`, `pygam` (only used by old univariate stub bias correction — verify before removing)
- MATLAB/CAT12 env vars (`MATLAB_BIN`, `SPM_PATH`, `CAT12_PATH`, `TIV_TEMPLATE`, `CAT12_*`)

`nibabel`/`nilearn` likely still needed for `visualization/brain.py`.

---

## 4. `.env` Changes

```bash
# Behavioral data (brainlink)
BRAINLINK_DB_PATH=/home/galkepler/Projects/brainlink/brainlink.db

# Neuroimaging tabular derivatives
TABULAR_DERIVATIVES_ROOT=/mnt/62/Processed_Data/derivatives/tabular
ATLAS_NAME=Schaefer2018N400n7Tian2020S2
ANAT_ATLASES=Schaefer2018N400n7,Tian2020S2
SESSION_VARIANT=cross   # cross | plain | subject

# REMOVE: SESSIONS_CSV, QUESTIONNAIRE_CSV, CAT12_*, QSIPARC_PATH, QSIRECON_PATH,
#         MATLAB_BIN, SPM_PATH, CAT12_PATH, TIV_TEMPLATE
```

---

## 5. Migration Phases

| Phase | Work | Output |
|---|---|---|
| 0 | Update `pyproject.toml` + `.env`, `uv sync` | brainlink + regional-stacker importable |
| 1 | `loaders/behavioral.py` + tests | sessions df with uid/subject_code/session_id/AGE/sex |
| 2 | `loaders/tabular_derivatives.py` (anat) + tests | long-format anatomical df incl. `tiv_mm3` |
| 3 | `loaders/tabular_derivatives.py` (dwi) + tests | long-format diffusion df |
| 4 | Rewrite `feature_store.py` metric lists + wide pivot | wide parquet per atlas/modality/metric |
| 5 | Rewrite `preprocessing/pipeline.py` + `config.py` | end-to-end `run()` against brainlink + tabular root |
| 6 | Remove obsolete loaders + deps + `.env` keys | dead code gone |
| 7 | `modeling/multivariate/estimator.py` using regional-stacker | regional BAGs via stacking |
| 8 | Update notebooks/scripts/CLI to new entry point | `python -m neuroalign.data.preprocessing.cli` works end-to-end |

---

## 6. Decisions (resolved)

1. **Atlas**: default `Schaefer2018N400n7Tian2020S2` (matches dwi naming). Anatomical = concat of `Schaefer2018N400n7` (cortex) + `Tian2020S2` (subcortex) folders. *(Note: tabular derivatives only have `Tian2020S2`, not the `S3` referenced in the stale `.env`.)*
2. **Session variant**: default `.cross` (maximizes session coverage — every scan gets a cross-sectional reconstruction), configurable via `PipelineConfig.session_variant` (`cross` | `plain` | `subject`).
3. **brainlink DB path**: read from `.env` (`BRAINLINK_DB_PATH`), no hardcoding.
4. **Region-name harmonization**: confirmed aligned across anat/dwi for this atlas — no mapping table needed.
5. **Connectome matrices**: confirmed out of scope for this restart.
