"""Thin FastAPI backend for the NiiVue Brain Explorer viewer (Step 5).

Serves the static viewer + assets and two JSON endpoints backed by
`neuroalign.session_data`, so the viewer can load any subject_code in the
cohort instead of a single hardcoded demo participant.

Run: `uv run python -m app.viewer_server` (serves on http://localhost:8000).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

_APP_DIR = Path(__file__).parent
_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
load_dotenv(_ROOT / ".env")

from neuroalign import session_data  # noqa: E402
from neuroalign.agent.region_narrator import narrate_region  # noqa: E402
from neuroalign.reference import load_region_reference  # noqa: E402

app = FastAPI(title="NeuroAlign Brain Explorer")


@app.get("/api/subject/{subject_code}")
def get_subject_sessions(subject_code: str) -> list[dict]:
    sessions = session_data.list_sessions(subject_code)
    if not sessions:
        if session_data.subject_known(subject_code):
            detail = "Subject exists but has no session in the regional BAG cohort."
        else:
            detail = "Subject code not found."
        raise HTTPException(status_code=404, detail=detail)
    return sessions


@app.get("/api/session/{uid}/{session_id}")
def get_session(uid: str, session_id: str) -> dict:
    try:
        return session_data.build_session(uid, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/interpretation/{uid}/{session_id}/{region_name}")
def get_region_interpretation(uid: str, session_id: str, region_name: str) -> dict:
    try:
        session = session_data.build_session(uid, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if region_name not in session["regions"]:
        raise HTTPException(status_code=404, detail=f"Unknown region: {region_name!r}")
    region = session["regions"][region_name]

    reference = load_region_reference()
    if region_name not in reference:
        raise HTTPException(status_code=404, detail=f"No reference data for region: {region_name!r}")
    info = reference[region_name]

    try:
        paragraph = narrate_region(
            plain_name=info["plain_name"],
            network=info.get("network"),
            structure=info["structure"],
            bag=region["bag"],
            percentile=region["percentile"],
            driver=region["driver"],
            deviant=region["deviant"],
            terms=tuple(t["term"] for t in info["terms"][:10]),
            age=session["meta"]["age"],
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"region": region_name, "paragraph": paragraph}


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/viewer/index.html")


# Mounted last: static routes must not shadow the /api/* routes above.
app.mount("/assets", StaticFiles(directory=_APP_DIR / "assets"), name="assets")
app.mount("/viewer", StaticFiles(directory=_APP_DIR / "viewer", html=True), name="viewer")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
