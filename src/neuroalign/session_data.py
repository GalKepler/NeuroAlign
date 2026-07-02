"""Builds enriched per-participant session data for the NiiVue viewer (Steps 4-5).

Reuses `BAGStore` to load the pre-computed `anat_baseline_demo` regional BAG /
contribution results, ranks each region against the full cohort to get a
percentile, and flags driver (top-|contribution|) and deviant (percentile
extreme) regions. Also resolves the human-facing `subject_code` (from
`metadata.parquet`) to the cohort's `(uid, session_id)` keys, for the viewer's
subject_code loader (`app/viewer_server.py`).

CLI: `python -m neuroalign.session_data` writes the hardcoded demo session to
`app/assets/demo_participant.json` (kept for the static-file fallback / smoke test).
"""

from __future__ import annotations

import functools
import json
import logging

import numpy as np
import pandas as pd

from neuroalign.modeling.bag_store import BAGStore
from neuroalign.modeling.result import BAGResult
from neuroalign.reference import APP_ASSETS_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

BAG_ROOT = REPO_ROOT / "data" / "processed_full"
RESULT_NAME = "multivariate/anat_baseline_demo"
METADATA_PATH = BAG_ROOT / "metadata.parquet"

DEMO_UID = "S909892"
DEMO_SESSION_ID = "202409041725"

N_DRIVERS = 10
DEVIANT_LOW_PCT = 5.0
DEVIANT_HIGH_PCT = 95.0


@functools.lru_cache(maxsize=1)
def _load_result() -> BAGResult:
    return BAGStore(BAG_ROOT).load_result(RESULT_NAME)


@functools.lru_cache(maxsize=1)
def _load_metadata() -> pd.DataFrame:
    return pd.read_parquet(METADATA_PATH)


def _percentiles(cohort: pd.DataFrame, session: pd.Series, region_cols: list[str]) -> dict[str, float]:
    """Percentile rank of `session`'s value within each region's cohort column."""
    out = {}
    for r in region_cols:
        col = cohort[r].to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        out[r] = float((col < session[r]).sum() / len(col) * 100.0)
    return out


def subject_known(subject_code: str) -> bool:
    """Whether subject_code exists anywhere in metadata (regardless of BAG cohort coverage)."""
    return bool((_load_metadata()["subject_code"] == subject_code).any())


def list_sessions(subject_code: str) -> list[dict]:
    """Sessions for a human-facing subject_code, sorted oldest -> newest.

    Returns `[]` if the code is unknown, or known but has no session in the
    `anat_baseline_demo` regional-BAG cohort (most metadata subjects aren't
    covered by this demo model fit - use `subject_known` to tell the two apart).
    """
    regional_bag = _load_result().regional_bag
    meta = _load_metadata()
    cohort_ids = regional_bag[["uid", "session_id"]]
    merged = cohort_ids.merge(meta, on=["uid", "session_id"], how="left")
    rows = merged[merged["subject_code"] == subject_code].sort_values("scan_date")
    return [
        {
            "uid": r["uid"],
            "session_id": r["session_id"],
            "scan_date": None if pd.isna(r["scan_date"]) else str(r["scan_date"]),
            "scan_tag": r["scan_tag"],
            "age": None if pd.isna(r["AGE"]) else float(r["AGE"]),
            "sex": r["sex"],
        }
        for _, r in rows.iterrows()
    ]


def build_session(uid: str, session_id: str) -> dict:
    result = _load_result()
    regional_bag = result.regional_bag
    regional_contribution = result.regional_contribution
    region_cols = [c for c in regional_bag.columns if c not in ("uid", "session_id")]

    mask = (regional_bag["uid"] == uid) & (regional_bag["session_id"] == session_id)
    if not mask.any():
        raise KeyError(f"Session not found in {RESULT_NAME}: uid={uid!r}, session_id={session_id!r}")
    session_bag = regional_bag[mask].iloc[0]
    session_contrib = regional_contribution[mask].iloc[0]

    pct = _percentiles(regional_bag, session_bag, region_cols)

    driver_regions = set(
        pd.Series({r: abs(session_contrib[r]) for r in region_cols}).sort_values(ascending=False).head(N_DRIVERS).index
    )

    overall_bag = float(result.bag[mask].iloc[0]["bag"])

    regions = {}
    for r in region_cols:
        p = pct[r]
        regions[r] = {
            "bag": float(session_bag[r]),
            "contribution": float(session_contrib[r]),
            "percentile": round(p, 1),
            "driver": r in driver_regions,
            "deviant": bool(p < DEVIANT_LOW_PCT or p > DEVIANT_HIGH_PCT),
        }

    meta_rows = _load_metadata()
    meta_row = meta_rows[(meta_rows["uid"] == uid) & (meta_rows["session_id"] == session_id)]
    meta = {"subject_code": None, "scan_date": None, "scan_tag": None, "age": None, "sex": None}
    if not meta_row.empty:
        m = meta_row.iloc[0]
        meta = {
            "subject_code": m["subject_code"],
            "scan_date": None if pd.isna(m["scan_date"]) else str(m["scan_date"]),
            "scan_tag": m["scan_tag"],
            "age": None if pd.isna(m["AGE"]) else float(m["AGE"]),
            "sex": m["sex"],
        }

    return {"uid": uid, "session_id": session_id, "bag": overall_bag, "meta": meta, "regions": regions}


def demo() -> None:
    """Self-check: percentiles in range, driver count matches, keys valid, subject lookup round-trips."""
    session = build_session(DEMO_UID, DEMO_SESSION_ID)
    regions = session["regions"]
    assert all(0.0 <= r["percentile"] <= 100.0 for r in regions.values()), "percentile out of [0,100]"
    assert sum(r["driver"] for r in regions.values()) == N_DRIVERS, "driver count mismatch"

    reference = json.loads((APP_ASSETS_DIR / "region_reference.json").read_text())
    assert set(regions.keys()) <= set(reference.keys()), "session has region keys missing from region_reference"

    subject_code = session["meta"]["subject_code"]
    assert subject_code, "demo session has no subject_code in metadata"
    sessions = list_sessions(subject_code)
    assert any(
        s["uid"] == DEMO_UID and s["session_id"] == DEMO_SESSION_ID for s in sessions
    ), "list_sessions(subject_code) doesn't include the demo session"

    # Guard the cache refactor: a second build_session call for the same session must
    # be identical (same cached result object, no state leakage across calls).
    session2 = build_session(DEMO_UID, DEMO_SESSION_ID)
    assert session2["regions"] == regions, "build_session not idempotent across calls"

    logger.info(
        "Self-check OK: subject_code=%s, %d sessions, %d regions, %d drivers, %d deviants",
        subject_code,
        len(sessions),
        len(regions),
        N_DRIVERS,
        sum(r["deviant"] for r in regions.values()),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demo()
    session = build_session(DEMO_UID, DEMO_SESSION_ID)
    out_path = APP_ASSETS_DIR / "demo_participant.json"
    out_path.write_text(json.dumps(session, indent=2))
    logger.info("Wrote %s (%d regions)", out_path, len(session["regions"]))


if __name__ == "__main__":
    main()
