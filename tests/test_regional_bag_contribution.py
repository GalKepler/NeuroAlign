"""Runnable check for the multivariate estimator's regional BAG / contribution matrices.

Verifies the linear-attribution identity the app viewer's per-region coloring
depends on: contribution[:, r] summed over regions + the meta-learner's
intercept must reproduce its scalar prediction exactly (see
`_compute_region_metrics` in `neuroalign.modeling.multivariate.estimator`).
"""

import numpy as np

from neuroalign.modeling._shared import compute_ipw_weights
from neuroalign.modeling.config import BAGConfig
from neuroalign.modeling.multivariate.estimator import _build_pipeline, _compute_region_metrics


def test_regional_contribution_sums_to_prediction_minus_intercept():
    rng = np.random.default_rng(0)
    n_sessions, n_regions, n_features_per_region = 60, 4, 3

    x = rng.normal(size=(n_sessions, n_regions * n_features_per_region))
    ages = rng.uniform(20, 80, size=n_sessions)
    region_mapping = {
        f"region_{r}": list(range(r * n_features_per_region, (r + 1) * n_features_per_region))
        for r in range(n_regions)
    }
    cfg = BAGConfig(n_splits=2, n_jobs=1, progress=False)

    _, oof, contribution, region_names = _compute_region_metrics(x, ages, region_mapping, cfg)

    assert oof.shape == (n_sessions, n_regions)
    assert contribution.shape == (n_sessions, n_regions)
    assert set(region_names) == set(region_mapping)

    # Independently refit the same full-data stacker to get ground-truth predictions.
    from regional_stacker import RegionalStackingRegressor

    weight = compute_ipw_weights(ages, cfg.ipw_bandwidth) if cfg.ipw else None
    n_subjects_proxy = max(2, min(5, n_sessions))
    stacker = RegionalStackingRegressor(
        region_mapping=region_mapping,
        base_estimator=_build_pipeline(),
        meta_estimator=_build_pipeline(),
        outer_cv=n_subjects_proxy,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        allow_nan=True,
    )
    stacker.fit(x, ages, sample_weight=weight)
    expected_pred = stacker.meta_estimator_.predict(stacker.oof_predictions_)
    intercept = stacker.meta_estimator_.named_steps["model"].intercept_

    computed_pred = contribution.sum(axis=1) + intercept
    np.testing.assert_allclose(computed_pred, expected_pred, atol=1e-8)


if __name__ == "__main__":
    test_regional_contribution_sums_to_prediction_minus_intercept()
    print("OK")
