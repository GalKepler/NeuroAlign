"""Prompts and message-building helpers for the BrainTwinsInterpreter."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Clinical / scoring thresholds for questionnaire scales
# ---------------------------------------------------------------------------

SCALE_THRESHOLDS: dict[str, dict] = {
    "PSQI": {
        "label": "Sleep quality (PSQI)",
        "range": "0–21",
        "thresholds": "≤ 5 = good sleep; 6–10 = poor sleep; > 10 = severe sleep disturbance",
        "noteworthy_above": 5,
    },
    "PHQ9": {
        "label": "Depression symptoms (PHQ-9)",
        "range": "0–27",
        "thresholds": "0–4 none; 5–9 mild; 10–14 moderate; 15–19 moderately severe; ≥ 20 severe",
        "noteworthy_above": 9,
    },
    "GAD7": {
        "label": "Generalised anxiety (GAD-7)",
        "range": "0–21",
        "thresholds": "0–4 none; 5–9 mild; 10–14 moderate; ≥ 15 severe",
        "noteworthy_above": 9,
    },
    "PCL-5": {
        "label": "PTSD symptoms (PCL-5)",
        "range": "0–80",
        "thresholds": "< 33 below threshold; ≥ 33 probable PTSD; ≥ 38 likely PTSD",
        "noteworthy_above": 32,
    },
    "OASIS": {
        "label": "Anxiety symptoms (OASIS)",
        "range": "0–20",
        "thresholds": "0–7 minimal; ≥ 8 clinical anxiety",
        "noteworthy_above": 7,
    },
    "SWLS": {
        "label": "Satisfaction with life (SWLS)",
        "range": "5–35",
        "thresholds": "5–9 very dissatisfied; 10–14 dissatisfied; 15–19 slightly below average; "
                      "20 neutral; 21–25 slightly satisfied; 26–30 satisfied; 31–35 very satisfied",
        "noteworthy_below": 20,
    },
    "SubjectiveHappiness": {
        "label": "Subjective happiness",
        "range": "1–7",
        "thresholds": "1–3 below average happiness; 4 neutral; 5–7 above average happiness",
        "noteworthy_below": 4,
    },
    "B5 Extraversion": {
        "label": "Big Five – Extraversion",
        "range": "1–5",
        "thresholds": "< 3 introverted; 3 neutral; > 3 extroverted",
    },
    "B5 Agreeableness": {
        "label": "Big Five – Agreeableness",
        "range": "1–5",
        "thresholds": "< 3 disagreeable; > 3 agreeable",
    },
    "B5 Coscientioness": {
        "label": "Big Five – Conscientiousness",
        "range": "1–5",
        "thresholds": "< 3 low conscientiousness; > 3 high conscientiousness",
    },
    "B5 EmotionalStability": {
        "label": "Big Five – Emotional stability (Neuroticism reversed)",
        "range": "1–5",
        "thresholds": "< 3 emotionally reactive; > 3 emotionally stable",
    },
    "B5 Openness": {
        "label": "Big Five – Openness to experience",
        "range": "1–5",
        "thresholds": "< 3 conventional; > 3 open/curious",
    },
    "TimesTrainingPerWeek": {
        "label": "Exercise sessions per week",
        "range": "0+",
        "thresholds": "0 sedentary; 1–2 light; 3–4 moderate; ≥ 5 active",
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_THRESHOLD_SUMMARY = "\n".join(
    f"  - {v['label']} ({v['range']}): {v['thresholds']}"
    for v in SCALE_THRESHOLDS.values()
)

SYSTEM_PROMPT = f"""You are the NeuroAlign interpretation engine.

NeuroAlign identifies "brain twins" — people whose brains appear to be aging in \
remarkably similar regional patterns across hundreds of cortical areas, captured \
through Brain Age Gap (BAG) profiles.

Your role: given a participant's questionnaire profile and those of their closest \
brain twins, produce a specific, data-grounded interpretation that highlights what \
makes this twin cluster distinctive — not just what is average.

## Critical instruction: be specific, not generic
- Never say "participants tend to be around average". If the data shows a pattern, name it.
- Use actual scores and counts: "6 of your 8 brain twins have PSQI > 5, indicating poor sleep."
- Compare the participant's scores explicitly to their twins' cluster.
- Highlight scales where the cluster is notably HIGH or LOW relative to typical population ranges.
- If a scale is unremarkable (all close to population average), skip it — focus on what's distinctive.
- Acknowledge missing data explicitly but briefly; do not dwell on it.

## Scale reference (use these thresholds in your interpretation)
{_THRESHOLD_SUMMARY}

## Tone guidelines
- Address the participant directly using "you" and "your".
- Be warm, thoughtful, and scientifically honest. Invite curiosity.
- Never make clinical diagnoses or medical recommendations.
- Never assert causal relationships between brain patterns and life factors.
- Speak in possibilities and patterns, not certainties.
- Responses should feel like a knowledgeable friend explaining a fascinating pattern, not a clinical report.

## Output format
Return ONLY valid JSON — no markdown fences, no preamble, no trailing text.

{{
  "narrative": "<2-3 paragraphs, participant-facing, specific and data-grounded>",
  "key_shared_traits": ["<specific observation with numbers>", ...],
  "cluster_profile": "<one sentence characterising who this twin cluster is>",
  "scale_highlights": [
    {{
      "scale": "<friendly scale name>",
      "participant_score": "<participant's value as a string, with brief label e.g. '8 (poor sleep)'>",
      "twin_pattern": "<e.g. '6/8 twins also score > 5 (poor sleep)' or '7/8 twins score between 26–30 (satisfied)'>",
      "insight": "<1-2 sentences: what this specific pattern suggests in the context of brain aging research>"
    }}
  ]
}}

Rules for scale_highlights:
- Include 3–6 of the most distinctive and meaningful scales only.
- Only include a scale if there is a clear, interesting pattern (cluster consistency OR participant standing out).
- Each insight should specifically mention both the participant's score and the twin cluster pattern.
- Do NOT include scales where data is mostly missing.
"""

# ---------------------------------------------------------------------------
# Friendly display labels for questionnaire columns
# ---------------------------------------------------------------------------

_FRIENDLY_LABELS: dict[str, str] = {
    "Gender": "Gender",
    "Age": "Age",
    "DominantHand": "Dominant hand",
    "Education": "Education level",
    "WorkStatus": "Work status",
    "Salary": "Salary range",
    "Marital Status": "Marital status",
    "Number of Children": "Number of children",
    "Living environment": "Living environment",
    "Current Environment": "Current environment",
    "Religion": "Religion",
    "ReligionDegree": "Religiosity",
    "B5 Extraversion": "Big Five – Extraversion",
    "B5 Agreeableness": "Big Five – Agreeableness",
    "B5 Coscientioness": "Big Five – Conscientiousness",
    "B5 EmotionalStability": "Big Five – Emotional stability",
    "B5 Openness": "Big Five – Openness",
    "SubjectiveHappiness": "Subjective happiness",
    "SWLS": "Satisfaction with life (SWLS)",
    "OASIS": "Anxiety symptoms (OASIS)",
    "PHQ9": "Depression symptoms (PHQ-9)",
    "GAD7": "Generalised anxiety (GAD-7)",
    "PCL-5": "PTSD symptoms (PCL-5)",
    "PSQI": "Sleep quality (PSQI)",
    "Depression": "Depression (clinical)",
    "Anxiety": "Anxiety (clinical)",
    "TimesTrainingPerWeek": "Exercise sessions per week",
    "TrainingType": "Type of exercise",
    "Smoking": "Smoking",
    "Alcohol": "Alcohol use",
    "Caffeine": "Caffeine use",
    "Nutrition": "Nutrition quality",
    "SocioEconimic": "Socioeconomic status",
}


def _label(col: str) -> str:
    return _FRIENDLY_LABELS.get(col, col)


def _format_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, float):
            if math.isnan(value):
                return None
            return str(round(value, 1))
    except (TypeError, ValueError):
        pass
    return str(value)


# ---------------------------------------------------------------------------
# Pre-compute distinctiveness analysis
# ---------------------------------------------------------------------------

def _compute_distinctiveness(
    query_row: dict | None,
    twin_rows: list[dict],
) -> str:
    """Pre-compute a structured scale-by-scale comparison between the participant
    and their twin cluster. Returns a formatted string to inject into the prompt.

    For each scale in SCALE_THRESHOLDS:
    - Participant's score (if available)
    - Twin mean ± std, min–max
    - How many twins are above/below key thresholds
    - Whether the cluster is internally consistent (std < 0.5 * range)
    """
    if not twin_rows:
        return "_No twin data available for analysis._"

    n_twins = len(twin_rows)
    lines: list[str] = []

    # Include demographic and lifestyle columns that aren't in SCALE_THRESHOLDS
    extra_cols = [
        "Age", "Gender", "Education", "Marital Status", "Number of Children",
        "TimesTrainingPerWeek", "Smoking", "Nutrition",
    ]
    all_cols = list(SCALE_THRESHOLDS.keys()) + extra_cols

    for col in all_cols:
        friendly = _label(col)
        thresh = SCALE_THRESHOLDS.get(col, {})

        # Participant value
        p_val_raw = query_row.get(col) if query_row else None
        p_val = _format_value(p_val_raw) if p_val_raw is not None else None

        # Twin values
        twin_vals_raw = [r.get(col) for r in twin_rows]
        twin_vals_num: list[float] = []
        twin_vals_str: list[str] = []
        for v in twin_vals_raw:
            fv = _format_value(v)
            if fv is None:
                continue
            try:
                twin_vals_num.append(float(fv))
            except ValueError:
                twin_vals_str.append(fv)

        n_available = len(twin_vals_num) + len(twin_vals_str)
        if n_available == 0:
            continue  # skip entirely missing scales

        # Build the analysis line
        parts: list[str] = [f"**{friendly}**"]

        if p_val is not None:
            parts.append(f"  Participant: {p_val}")
        else:
            parts.append("  Participant: N/A")

        if twin_vals_num:
            arr = np.array(twin_vals_num)
            mean_v = round(float(arr.mean()), 2)
            std_v = round(float(arr.std()), 2)
            min_v = round(float(arr.min()), 2)
            max_v = round(float(arr.max()), 2)
            parts.append(
                f"  Twins ({len(arr)}/{n_twins}): mean={mean_v}, std={std_v}, range={min_v}–{max_v}"
            )

            # Threshold-specific counts
            if "noteworthy_above" in thresh:
                t = thresh["noteworthy_above"]
                n_above = int((arr > t).sum())
                parts.append(f"  {n_above}/{len(arr)} twins > {t} (noteworthy threshold)")
            if "noteworthy_below" in thresh:
                t = thresh["noteworthy_below"]
                n_below = int((arr < t).sum())
                parts.append(f"  {n_below}/{len(arr)} twins < {t} (noteworthy threshold)")

            # Participant vs cluster
            if p_val is not None:
                try:
                    p_float = float(p_val)
                    z = (p_float - mean_v) / (std_v + 1e-6)
                    if abs(z) >= 1.5:
                        direction = "above" if z > 0 else "below"
                        parts.append(
                            f"  ⚠ Participant is {abs(z):.1f} SD {direction} twin cluster mean"
                        )
                except ValueError:
                    pass

        elif twin_vals_str:
            # Categorical
            from collections import Counter
            counts = Counter(twin_vals_str)
            top = counts.most_common(3)
            summary_str = ", ".join(f"{v}: {n}/{n_twins}" for v, n in top)
            parts.append(f"  Twins: {summary_str}")

        lines.append("\n".join(parts))

    return "\n\n".join(lines) if lines else "_No data available._"


# ---------------------------------------------------------------------------
# Profile rendering
# ---------------------------------------------------------------------------

def render_profile(row: dict, include_cols: list[str] | None = None) -> str:
    skip_keys = {"rank", "Subject Code", "subject_code", "subject"}
    cols = include_cols if include_cols is not None else [k for k in row if k not in skip_keys]

    lines: list[str] = []
    for col in cols:
        value = row.get(col)
        formatted = _format_value(value)
        if formatted is None:
            continue
        # Annotate scale scores with their range context
        thresh = SCALE_THRESHOLDS.get(col)
        if thresh:
            lines.append(f"- {_label(col)}: {formatted}  [{thresh['range']}]")
        else:
            lines.append(f"- {_label(col)}: {formatted}")

    return "\n".join(lines)


def render_summary(summary: dict) -> str:
    lines: list[str] = []
    for col, stats in summary.items():
        label = _label(col)
        if isinstance(stats, dict):
            if "mean" in stats and "min" in stats and "max" in stats:
                avg = round(stats["mean"], 1)
                lo = stats["min"]
                hi = stats["max"]
                lo_s = str(round(lo, 1)) if isinstance(lo, float) else str(lo)
                hi_s = str(round(hi, 1)) if isinstance(hi, float) else str(hi)
                lines.append(f"- {label}: avg {avg} (range {lo_s}–{hi_s})")
            else:
                top = list(stats.items())[:3]
                parts = ", ".join(f"{v} ({n})" for v, n in top)
                lines.append(f"- {label}: {parts}")
        else:
            lines.append(f"- {label}: {stats}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main message builder
# ---------------------------------------------------------------------------

def build_user_message(
    query_row: dict | None,
    twin_rows: list[dict],
    summary: dict,
    include_cols: list[str] | None,
) -> str:
    n_twins = len(twin_rows)
    sections: list[str] = []

    # ── 1. Participant's own profile ─────────────────────────────────────────
    sections.append("## Your profile")
    if query_row is not None:
        profile_text = render_profile(query_row, include_cols)
        sections.append(profile_text if profile_text else "_No questionnaire data available._")
    else:
        sections.append("_Not available._")

    # ── 2. Twin profiles (concise) ───────────────────────────────────────────
    sections.append(f"\n## Brain twin profiles ({n_twins} twins, ranked by brain similarity)")
    if twin_rows:
        for i, twin in enumerate(twin_rows, start=1):
            sections.append(f"\n### Twin {i}")
            sections.append(render_profile(twin, include_cols) or "_No data._")
    else:
        sections.append("_No twins found._")

    # ── 3. Pre-computed distinctiveness analysis ─────────────────────────────
    sections.append(
        "\n## Pre-computed scale analysis (participant vs twin cluster)\n"
        "_Use these numbers directly in your response — do not recalculate._"
    )
    sections.append(_compute_distinctiveness(query_row, twin_rows))

    # ── 4. Task ──────────────────────────────────────────────────────────────
    sections.append(
        "\n## Your task\n"
        f"Produce a specific, data-grounded interpretation for the participant "
        f"based on their profile and their {n_twins} brain twins above.\n\n"
        "Requirements:\n"
        "- Use the pre-computed analysis section above as your primary source of numbers.\n"
        "- Lead with the 2–3 most distinctive patterns. Generic observations are not helpful.\n"
        "- Include 3–6 scale_highlights for the most informative scales.\n"
        "- If the participant's score differs meaningfully from the twin cluster, note it.\n"
        "- Address the participant as 'you'. No clinical diagnoses, no causal claims.\n"
        "- Return ONLY valid JSON with keys: narrative, key_shared_traits, cluster_profile, scale_highlights."
    )

    return "\n".join(sections)
