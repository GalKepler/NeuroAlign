# Regional BAG Estimation - Implementation Plan

## Overview

Implement univariate regional BAG estimation: for each brain region independently, predict chronological age from that region's metric value (+ covariates), then compute BAG = predicted_age - actual_age. Apply post-hoc bias correction per region.

Structured to separate shared BAG infrastructure (config, result, bias correction) from the univariate-specific logic, so a future `neuroalign.modeling.multivariate` can reuse the same foundations.

## File Structure

```
src/neuroalign/modeling/
├── __init__.py              # Top-level exports (edit)
├── config.py                # BAGConfig (new, shared)
├── result.py                # BAGResult dataclass (new, shared)
├── univariate/
│   ├── __init__.py          # Univariate exports (new)
│   └── estimator.py         # RegionalBAGEstimator (new)
```

## Files to Create/Edit

### 1. `src/neuroalign/modeling/config.py` (new, shared)

Pydantic config model used by both univariate and (future) multivariate estimators:

```python
class BAGConfig(BaseModel):
    n_splits: int = 5                    # GroupKFold splits
    model_type: str = "ridge"            # "ridge" | "xgboost" | "lightgbm"
    polynomial_degree: int = 2           # polynomial features for metric values
    bias_correction: bool = True         # post-hoc de Lange & Cole correction
    ipw: bool = True                     # inverse probability weighting
    ipw_bandwidth: float = 2.0           # KDE bandwidth for IPW
    random_state: int = 42
    n_jobs: int = 1                      # parallel region fitting
    progress: bool = True                # tqdm progress bars

    # Column name mapping
    age_col: str = "age"
    sex_col: str = "sex"
    tiv_col: str = "tiv"
    subject_col: str = "subject_code"
    session_col: str = "session_id"
```

### 2. `src/neuroalign/modeling/result.py` (new, shared)

Result container reusable by any BAG estimation approach:

```python
@dataclass
class BAGResult:
    bag: pd.DataFrame              # corrected BAG (sessions x regions, wide)
    bag_uncorrected: pd.DataFrame  # raw BAG before correction
    predicted_age: pd.DataFrame    # predicted brain age per region
    region_metrics: pd.DataFrame   # per-region R^2, MAE, correlation
    config: BAGConfig

    def save(self, output_dir: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "BAGResult": ...
```

### 3. `src/neuroalign/modeling/univariate/estimator.py` (new)

The univariate-specific logic:

```python
class RegionalBAGEstimator:
    def __init__(self, config: BAGConfig): ...

    def fit_predict(
        self,
        features: pd.DataFrame,    # wide: [subject_col, session_col, region_1, ..., region_N]
        metadata: pd.DataFrame,     # [subject_col, session_col, age_col, sex_col, tiv_col]
    ) -> BAGResult: ...

    @staticmethod
    def long_to_wide(
        long_df: pd.DataFrame,
        metric_col: str,
        region_col: str = "label",
        id_cols: list[str] | None = None,
    ) -> pd.DataFrame: ...
```

Private helper functions in the same module:
- `_compute_ipw_weights(ages, bandwidth)` - KDE-based inverse probability weighting
- `_apply_bias_correction(bag_df, ages)` - post-hoc linear correction per region
- `_build_model(config)` - model factory (Ridge/XGBoost/LightGBM)

### 4. `src/neuroalign/modeling/univariate/__init__.py` (new)

Export `RegionalBAGEstimator`.

### 5. `src/neuroalign/modeling/__init__.py` (edit)

Export shared pieces + univariate estimator:
```python
from .config import BAGConfig
from .result import BAGResult
from .univariate import RegionalBAGEstimator
```

## Algorithm (`fit_predict`)

1. **Merge & validate**: Join features with metadata on subject_col + session_col. Drop rows with missing age/sex/tiv. Encode sex as 0/1.

2. **Identify regions**: All columns in features except subject_col, session_col.

3. **Group-aware K-fold**: `GroupKFold(n_splits)` with groups = subject_col. Ensures a subject's repeated sessions stay in the same fold.

4. **For each fold** (train/test split):
   - **Compute IPW weights** (training set only): KDE on training ages, weights = 1/density, normalized to mean=1.
   - **For each region**:
     - Build feature matrix: `[region_metric, sex_encoded, tiv]`
     - For ridge: apply `PolynomialFeatures(degree)` to the metric column only
     - For tree models: skip polynomial features (trees handle non-linearity natively)
     - Fit model with `sample_weight=ipw_weights`
     - Predict on test set
   - Store all test-set predictions

5. **Assemble**: Concatenate test predictions across folds. Compute `BAG = predicted_age - actual_age` per region.

6. **Bias correction** (per region independently):
   - Fit: `BAG_r = alpha * age + beta` (OLS)
   - `BAG_corrected_r = BAG_r - (alpha * age + beta)`

7. **Compute metrics** (per region):
   - R^2 (predicted vs actual age)
   - MAE
   - Pearson correlation

8. **Return** `BAGResult`.

## Output Format

Saved under `output_dir/`:
```
bag.parquet              # sessions x regions (corrected BAG values)
bag_uncorrected.parquet  # sessions x regions (raw BAG)
predicted_age.parquet    # sessions x regions (predicted brain age)
region_metrics.parquet   # per-region: r2, mae, correlation
config.json              # BAGConfig used
```

All wide parquet files have `subject_code` and `session_id` as the first two columns, followed by one column per region.

## Usage Example (notebook)

```python
from neuroalign.modeling import RegionalBAGEstimator, BAGConfig

config = BAGConfig(model_type="ridge", n_splits=5, bias_correction=True)
estimator = RegionalBAGEstimator(config)

# Convert long format to wide
features_wide = RegionalBAGEstimator.long_to_wide(metric_long, metric_col="volume_mm3")

# Build metadata DataFrame
meta = metadata[["subject_code", "session_id", "Age@Scan", "Gender"]].copy()
meta = meta.merge(tiv_df, on=["subject_code", "session_id"])
meta = meta.rename(columns={"Age@Scan": "age", "Gender": "sex"})

# Compute BAGs
result = estimator.fit_predict(features_wide, meta)
result.save(Path("data/processed/bag/anatomical_gm_volume"))
```

## Future Extension (multivariate)

```
src/neuroalign/modeling/
├── config.py                # Shared BAGConfig
├── result.py                # Shared BAGResult
├── univariate/              # <-- current work
│   └── estimator.py
└── multivariate/            # <-- future
    └── estimator.py         # Uses all regions jointly to predict age
```

The multivariate estimator would reuse `BAGConfig`, `BAGResult`, and the same bias correction / IPW logic, but train a single model with all regions as features instead of looping per-region.

## Verification

1. Run in notebook with `anatomical_gm` / `volume_mm3` data
2. Check `result.region_metrics` - R^2 values should be non-trivial for regions with strong age effects (notebook showed R^2 up to ~0.72)
3. Verify bias correction: correlation of corrected BAG with age should be near zero
4. Verify saved parquet files load back via `BAGResult.load()`
