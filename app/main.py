"""NeuroAlign — participant-facing brain profile app."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ── Path setup (works whether run from project root or app/) ─────────────────
_APP_DIR = Path(__file__).parent
_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_ROOT / "src"))
load_dotenv(_ROOT / ".env")

from neuroalign.agent import (  # noqa: E402
    BrainTwinsInterpreter,
    InterpreterConfig,
    RegionalProfileInterpreter,
)
from neuroalign.data.questionnaire import QuestionnaireLookup  # noqa: E402
from neuroalign.embedding.result import EmbeddingResult  # noqa: E402
from neuroalign.retrieval.retriever import BrainRetriever  # noqa: E402

# ── Paths from .env ───────────────────────────────────────────────────────────
_MODELS_DIR = _ROOT / os.getenv("MODELS_DIR", "models")
_EMBEDDING_DIR = _MODELS_DIR / "embedding_retrieval/embedding"
_RETRIEVER_DIR = _MODELS_DIR / "embedding_retrieval/retriever"
_QUESTIONNAIRE_CSV = Path(
    os.path.expanduser(os.getenv("QUESTIONNAIRE_CSV", "~/Downloads/qcenter.csv"))
)
_ATLAS_PARQUET = _ROOT / "data/processed/long/anatomical_ct.parquet"
_TIAN_ATLAS = os.getenv(
    "TIAN_ATLAS_PATH",
    "/media/storage/Projects/roi-summarization-fallacy/notebooks/tutorial_data/tian_atlas/surfaces",
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroAlign — Your Brain Profile",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-label { font-size: 0.75rem !important; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached resource loading ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_all() -> dict:
    config = InterpreterConfig(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        max_tokens=15000,
        temperature=0.7,
    )
    return {
        "embedding": EmbeddingResult.load(_EMBEDDING_DIR),
        "retriever": BrainRetriever.load(_RETRIEVER_DIR),
        "questionnaire": QuestionnaireLookup.from_csv(_QUESTIONNAIRE_CSV),
        "twins_interp": BrainTwinsInterpreter(config=config),
        "regional_interp": RegionalProfileInterpreter.from_parquet(_ATLAS_PARQUET, config=config),
        "config": config,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_sessions(embedding: EmbeddingResult, subject: str) -> list[str]:
    df = embedding.vectors
    mask = df[embedding.config.subject_col] == subject
    return sorted(df.loc[mask, embedding.config.session_col].unique().tolist())


# ── Demographics / personality columns shown in twin cards ───────────────────
_TWIN_DEMO_COLS = [
    ("Gender", "Gender"),
    ("Age", "Age"),
    ("Education", "Education"),
    ("Marital Status", "Marital status"),
    ("Number of Children", "Children"),
    ("B5 Extraversion", "Extraversion"),
    ("B5 Agreeableness", "Agreeableness"),
    ("B5 Coscientioness", "Conscientiousness"),
    ("B5 EmotionalStability", "Emot. stability"),
    ("B5 Openness", "Openness"),
    ("SubjectiveHappiness", "Happiness"),
    ("SWLS", "Life satisfaction"),
    ("TimesTrainingPerWeek", "Training/week"),
    ("Smoking", "Smoking"),
    ("Nutrition", "Diet"),
]

# ── Validated clinical questionnaire scales ───────────────────────────────────
# Each entry: (data_col, short_label, display_label)
_CLINICAL_COLS = [
    ("PHQ9",  "PHQ-9",  "Depression (PHQ-9)"),
    ("GAD7",  "GAD-7",  "Gen. Anxiety (GAD-7)"),
    ("PCL-5", "PCL-5",  "PTSD (PCL-5)"),
    ("OASIS", "OASIS",  "Anxiety (OASIS)"),
    ("PSQI",  "PSQI",   "Sleep (PSQI)"),
]


def _clinical_label(col: str, value: float) -> str:
    """Return a score string annotated with its clinical severity category."""
    v = float(value)
    if col == "PHQ9":
        if v <= 4:   cat = "Minimal"
        elif v <= 9: cat = "Mild"
        elif v <= 14: cat = "Moderate"
        elif v <= 19: cat = "Mod. severe"
        else:        cat = "Severe"
    elif col == "GAD7":
        if v <= 4:   cat = "Minimal"
        elif v <= 9: cat = "Mild"
        elif v <= 14: cat = "Moderate"
        else:        cat = "Severe"
    elif col == "PCL-5":
        if v < 33:   cat = "Below threshold"
        elif v < 38: cat = "Probable PTSD"
        else:        cat = "Likely PTSD"
    elif col == "OASIS":
        if v <= 7:   cat = "Minimal"
        else:        cat = "Clinical"
    elif col == "PSQI":
        if v <= 5:   cat = "Good sleep"
        elif v <= 10: cat = "Poor sleep"
        else:        cat = "Severe disturbance"
    else:
        return str(round(v, 1))
    return f"{v:.0f} — {cat}"


def render_clinical_scores(row: dict, n_cols: int = 5) -> None:
    """Render validated clinical scale scores as labelled metric cards."""
    items = []
    for col_key, short_label, display_label in _CLINICAL_COLS:
        val = row.get(col_key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            items.append((display_label, _clinical_label(col_key, val)))
        except (TypeError, ValueError):
            items.append((display_label, str(val)))

    if not items:
        return
    cols = st.columns(n_cols)
    for i, (label, display) in enumerate(items):
        cols[i % n_cols].metric(label, display)


def render_twin_card(row: dict, rank: int, distance: float | None) -> None:
    """Render one brain twin as a compact card: demographics + clinical scores."""
    header = f"🧬 Brain twin #{rank}"
    if distance is not None:
        header += f"&nbsp;&nbsp;·&nbsp;&nbsp;<small>similarity score: `{distance:.3f}`</small>"

    with st.container(border=True):
        st.markdown(f"**{header}**", unsafe_allow_html=True)

        # Row 1 — demographics, personality, lifestyle (up to 10 fields)
        demo_cols = st.columns(5)
        shown = 0
        for col_key, label in _TWIN_DEMO_COLS:
            val = row.get(col_key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, float):
                val = round(val, 1)
            demo_cols[shown % 5].metric(label, val)
            shown += 1
            if shown >= 10:
                break

        # Row 2 — validated clinical scales (always shown when present)
        has_clinical = any(
            row.get(col) is not None
            and not (isinstance(row.get(col), float) and pd.isna(row.get(col)))
            for col, _, __ in _CLINICAL_COLS
        )
        if has_clinical:
            st.markdown(
                "<small>**Clinical scales** ·  "
                "PHQ-9: depression · GAD-7: anxiety · "
                "PCL-5: PTSD · OASIS: anxiety severity · PSQI: sleep quality</small>",
                unsafe_allow_html=True,
            )
            render_clinical_scores(row)


def render_bar_chart(stats_df: pd.DataFrame) -> None:
    """Render side-by-side cortical / subcortical BAG bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, is_sub, title in zip(
        axes,
        [False, True],
        ["Cortical Networks", "Subcortical Structures"],
    ):
        sub = stats_df[stats_df["is_subcortical"] == is_sub].sort_values("mean_z")
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = ["#E74C3C" if z > 0 else "#3498DB" for z in sub["mean_z"]]
        ax.barh(sub["display_name"], sub["mean_z"], color=colors, height=0.6)
        ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean BAG z-score", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    older_p = mpatches.Patch(color="#E74C3C", label="Appears older ▲")
    younger_p = mpatches.Patch(color="#3498DB", label="Appears younger ▼")
    fig.legend(
        handles=[older_p, younger_p],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    st.pyplot(fig)
    plt.close(fig)


def render_scale_highlights(highlights: list[dict]) -> None:
    """Render per-scale LLM insights as structured cards."""
    if not highlights:
        return
    st.markdown("**📊 Scale-by-scale insights**")
    for h in highlights:
        scale    = h.get("scale", "")
        p_score  = h.get("participant_score", "—")
        twin_pat = h.get("twin_pattern", "—")
        insight  = h.get("insight", "")
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.2, 1.8, 3])
            c1.markdown(f"**{scale}**")
            c2.metric("Your score", p_score)
            c3.markdown(f"**Twin cluster:** {twin_pat}")
            if insight:
                st.caption(insight)


def render_brain_map(subject_code: str, session_id: str, embedding: EmbeddingResult) -> None:
    """Render static yabplot brain image + interactive nilearn surface view."""
    from neuroalign.visualization.brain import render_brain_png, render_interactive_html

    z_vector = embedding.get_vector(subject_code, session_id)

    # ── Static brain image (yabplot) ─────────────────────────────────────────
    _key_png = f"brain_png_{subject_code}_{session_id}"
    if _key_png not in st.session_state:
        with st.spinner("Rendering brain map… (first load takes ~20 s)"):
            try:
                st.session_state[_key_png] = render_brain_png(
                    z_vector,
                    tian_atlas_path=_TIAN_ATLAS,
                )
            except Exception as exc:
                st.session_state[_key_png] = None
                st.warning(f"Brain image rendering failed: {exc}")

    if st.session_state.get(_key_png):
        st.image(
            st.session_state[_key_png],
            caption="Regional BAG z-scores across 400 cortical + 50 subcortical parcels. "
                    "Red = appears older · Blue = appears younger.",
            use_container_width=True,
        )

    # ── Interactive 3D brain (nilearn) ────────────────────────────────────────
    with st.expander("🔄 Interactive 3D brain — click & drag to rotate", expanded=False):
        _key_html = f"brain_html_{subject_code}_{session_id}"
        if _key_html not in st.session_state:
            with st.spinner("Building interactive surface…"):
                try:
                    lh_html, rh_html = render_interactive_html(z_vector)
                    st.session_state[_key_html] = (lh_html, rh_html)
                except Exception as exc:
                    st.session_state[_key_html] = None
                    st.warning(f"Interactive surface failed: {exc}")

        brain_html = st.session_state.get(_key_html)
        if brain_html:
            lh_html, rh_html = brain_html
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Left hemisphere**")
                components.html(lh_html, height=420, scrolling=False)
            with col2:
                st.markdown("**Right hemisphere**")
                components.html(rh_html, height=420, scrolling=False)
            st.caption(
                "🔴 Red = appears older than typical for your age group · "
                "🔵 Blue = appears younger / more preserved. "
                "Gray areas = no data for this parcel."
            )


# ── Load models (once, cached) ────────────────────────────────────────────────
try:
    models = load_all()
except Exception as exc:
    st.error(
        f"**Failed to load models.** Check that the embedding and retriever are built.\n\n`{exc}`"
    )
    st.stop()

embedding = models["embedding"]
retriever = models["retriever"]
questionnaire = models["questionnaire"]
twins_interp = models["twins_interp"]
regional_interp = models["regional_interp"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://i.imgur.com/placeholder.png", width=60) if False else None
    st.title("🧠 NeuroAlign")
    st.caption("SNBB Brain Bank — Participant Portal")
    st.divider()

    subject_code = st.text_input(
        "Subject code",
        placeholder="e.g. 1234",
        help="Enter the subject code you received from the study team.",
    ).strip()

    session_id: str | None = None
    if subject_code:
        sessions = get_sessions(embedding, subject_code)
        if not sessions:
            st.warning("Subject code not found. Double-check and try again.")
        elif len(sessions) == 1:
            session_id = sessions[0]
            st.success(f"Session: `{session_id}`")
        else:
            session_id = st.selectbox("Session", sessions, index=len(sessions) - 1)

    st.divider()
    n_twins = st.slider("Brain twins to show", min_value=10, max_value=100, value=25)
    st.caption("Higher = more diverse matches, lower = most similar only.")

# ── Landing ───────────────────────────────────────────────────────────────────
if not subject_code or not session_id:
    st.title("Welcome to NeuroAlign")
    st.markdown(
        """
        **NeuroAlign** maps how different regions of your brain are aging compared
        to other participants in the study — and finds your *brain twins*: people
        whose brains show a remarkably similar pattern to yours.

        👈 **Enter your subject code in the sidebar to get started.**
        """
    )
    st.stop()
else:
    # ── Retrieval ─────────────────────────────────────────────────────────────
    _cache_key_result = f"retrieval_{subject_code}_{session_id}_{n_twins}"
    if _cache_key_result not in st.session_state:
        with st.spinner("Finding your brain twins…"):
            try:
                original_k = retriever.config.k
                retriever.config.k = max(n_twins, original_k)
                _ret = retriever.query(subject_code, session_id)
                retriever.config.k = original_k
                _ret.matches = _ret.matches[_ret.matches["rank"] <= n_twins].reset_index(drop=True)
                st.session_state[_cache_key_result] = _ret
            except KeyError:
                st.error(
                    f"Subject `{subject_code}` / session `{session_id}` not found in the index. "
                    "This session may not have been processed yet."
                )
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")

    result = st.session_state.get(_cache_key_result)
    if result is None:
        st.stop()
    else:
        profiles = questionnaire.get_profiles(result)
        summary = questionnaire.summarise(profiles)

        # ── Header ────────────────────────────────────────────────────────────
        st.title("Your Brain Profile")
        st.caption(f"Subject `{subject_code}` · Session `{session_id}`")

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab_twins, tab_regional = st.tabs(["🧬 Brain Twins", "🧠 Brain Profile"])

        # ═════════════════════════════════════════════════════════════════════
        # TAB 1 — BRAIN TWINS
        # ═════════════════════════════════════════════════════════════════════
        with tab_twins:
            st.markdown(
                "These participants' brains show a similar **regional aging pattern** to yours — "
                "similar Brain Age Gap values across 450+ cortical and subcortical regions."
            )

            # ── Participant's own clinical scores ─────────────────────────────
            query_row_df = profiles[profiles["rank"] == 0]
            if not query_row_df.empty:
                q = query_row_df.iloc[0].to_dict()
                has_clinical = any(
                    q.get(col) is not None
                    and not (isinstance(q.get(col), float) and pd.isna(q.get(col)))
                    for col, *_ in _CLINICAL_COLS
                )
                if has_clinical:
                    with st.expander("🩺 Your clinical questionnaire scores", expanded=True):
                        st.caption(
                            "Your scores on validated psychological scales, shown here for context "
                            "alongside your brain twins."
                        )
                        render_clinical_scores(q, n_cols=5)
                        st.caption(
                            "PHQ-9: depression (0–27) · GAD-7: generalised anxiety (0–21) · "
                            "PCL-5: PTSD symptoms (0–80) · OASIS: anxiety severity (0–20) · "
                            "PSQI: sleep quality (0–21, higher = worse)"
                        )
                    st.divider()

            # ── Brain twin cards ──────────────────────────────────────────────
            twin_rows = profiles[profiles["rank"] > 0].sort_values("rank")
            dist_map = result.matches.set_index("rank")["distance"].to_dict()

            for _, row in twin_rows.iterrows():
                rank = int(row["rank"])
                render_twin_card(row.to_dict(), rank, dist_map.get(rank))

            st.divider()

            # ── LLM interpretation ────────────────────────────────────────────
            st.subheader("💬 What your brain twins have in common")
            _cache_key_twins = f"twins_interp_{subject_code}_{session_id}_{n_twins}"
            if _cache_key_twins not in st.session_state:
                with st.spinner("Asking the AI…"):
                    try:
                        st.session_state[_cache_key_twins] = twins_interp.interpret(
                            result, profiles, summary
                        )
                    except ValueError as exc:
                        st.warning(str(exc))
                        st.session_state[_cache_key_twins] = None
                    except Exception as exc:
                        st.error(f"Interpretation failed: {exc}")
                        st.session_state[_cache_key_twins] = None

            interp = st.session_state[_cache_key_twins]
            if interp:
                col_l, col_r = st.columns([1, 2])
                with col_l:
                    st.markdown("**🧬 Cluster profile**")
                    st.info(interp.cluster_profile or "—")
                    st.markdown("**🔍 Shared traits**")
                    for trait in interp.key_shared_traits:
                        st.markdown(f"- {trait}")
                with col_r:
                    st.markdown("**📖 Your story**")
                    st.markdown(interp.narrative)

                if interp.scale_highlights:
                    st.divider()
                    render_scale_highlights(interp.scale_highlights)

        # ═════════════════════════════════════════════════════════════════════
        # TAB 2 — REGIONAL BAG PROFILE
        # ═════════════════════════════════════════════════════════════════════
        with tab_regional:
            st.markdown(
                "How are different brain networks aging relative to others your age in the study? "
                "🔴 **Red** = appears older than typical · 🔵 **Blue** = appears younger / more preserved."
            )

            _cache_key_reg = f"regional_{subject_code}_{session_id}"
            if _cache_key_reg not in st.session_state:
                with st.spinner("Computing regional profile…"):
                    try:
                        st.session_state[_cache_key_reg] = regional_interp.interpret(
                            embedding, subject_code, session_id
                        )
                    except Exception as exc:
                        st.error(f"Regional profile failed: {exc}")
                        st.session_state[_cache_key_reg] = None

            reg = st.session_state[_cache_key_reg]
            if reg:
                render_bar_chart(reg.to_stats_df())

                st.divider()

                st.subheader("🧠 Brain Map")
                render_brain_map(subject_code, session_id, embedding)

                st.divider()

                st.subheader("💬 What this means")
                st.info(reg.overall_summary or "—")

                if reg.network_highlights:
                    st.markdown("**Network highlights**")
                    for h in reg.network_highlights:
                        direction = (
                            "▲ appears older" if h.get("direction") == "older" else "▼ appears younger"
                        )
                        st.markdown(
                            f"- **{h.get('network', '')}** ({direction}): {h.get('description', '')}"
                        )

                st.markdown("---")
                st.markdown(reg.narrative)
