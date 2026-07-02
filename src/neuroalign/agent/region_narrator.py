"""Single-region BAG interpretation: one region's z/BAG value + Neurosynth terms -> paragraph.

Used by the viewer's per-region panel (`app/viewer_server.py`) for an
on-demand "explain this region" call. Distinct from `regional_interpreter.py`,
which summarizes a participant's *whole* regional profile at the network
level; this module explains a single clicked region using its own BAG value
plus the Neurosynth literature already merged into `region_reference.json`.
"""
from __future__ import annotations

import functools
import logging
import os

from neuroalign.agent.config import InterpreterConfig

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the NeuroAlign region explainer. A research participant clicked on a single "
    "brain region in an interactive viewer. You are given that region's Brain Age Gap (BAG) "
    "for this participant and the Neurosynth meta-analytic terms most associated with its "
    "coordinates (from thousands of published fMRI studies).\n\n"
    "Brain Age Gap (BAG) is the difference between how old a region's structure appears and "
    "the participant's actual age. Positive = appears structurally older than typical for "
    "their age; negative = appears younger / more preserved.\n\n"
    "Write ONE short paragraph (3-5 sentences) for the participant, in plain language:\n"
    "1. State what this region does, grounded in the given Neurosynth terms.\n"
    "2. State the participant's BAG value and cohort percentile for this region, and what "
    "that means in context (e.g. an outlier vs. typical).\n"
    "3. Offer a measured, research-context interpretation of what this pattern might suggest.\n\n"
    "Rules: address the participant as 'you'. No clinical diagnoses, no medical advice, no "
    "causal claims. This is a research pattern, not a clinical finding -- say so if the value "
    "is extreme. Do not repeat raw term lists verbatim; weave them into plain description. "
    "Return ONLY the paragraph text -- no headers, no markdown, no JSON."
)


def _build_user_message(
    plain_name: str,
    network: str | None,
    structure: str,
    bag: float,
    percentile: float,
    driver: bool,
    deviant: bool,
    terms: tuple[str, ...],
    age: float | None,
) -> str:
    lines = [
        f"Region: {plain_name} ({structure}{f', {network} network' if network else ''})",
        f"Neurosynth terms most associated with this region's coordinates: {', '.join(terms) or 'none available'}",
        f"Participant BAG for this region: {bag:+.2f} years",
        f"Cohort percentile: {percentile:.1f}",
    ]
    if age is not None:
        lines.append(f"Participant age: {age:.1f}")
    if driver:
        lines.append("This region is among the top contributors to the participant's overall BAG.")
    if deviant:
        lines.append("This region's percentile is extreme (below 5th or above 95th) relative to the cohort.")
    return "\n".join(lines)


class RegionNarrator:
    """Generates a single-region interpretation paragraph via the Gemini API."""

    def __init__(self, config: InterpreterConfig | None = None) -> None:
        self.config = config or InterpreterConfig(max_tokens=400)
        self._client = None

    def _get_client(self):
        if self._client is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY is not set. Get a free key at aistudio.google.com."
                )
            try:
                from google import genai  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("google-genai is required: pip install google-genai") from exc
            self._client = genai.Client(api_key=api_key)
        return self._client

    def narrate(
        self,
        plain_name: str,
        network: str | None,
        structure: str,
        bag: float,
        percentile: float,
        driver: bool,
        deviant: bool,
        terms: tuple[str, ...],
        age: float | None = None,
    ) -> str:
        """Return a plain-text paragraph interpreting one region for one participant."""
        from google.genai import types as genai_types  # noqa: PLC0415

        user_message = _build_user_message(
            plain_name, network, structure, bag, percentile, driver, deviant, terms, age
        )
        client = self._get_client()
        response = client.models.generate_content(
            model=self.config.model,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                # A short, single-paragraph task doesn't need reasoning tokens, and
                # thinking eats into max_output_tokens on 2.5-flash and truncates output.
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()


@functools.lru_cache(maxsize=1)
def _default_narrator() -> RegionNarrator:
    return RegionNarrator()


@functools.lru_cache(maxsize=512)
def narrate_region(
    plain_name: str,
    network: str | None,
    structure: str,
    bag: float,
    percentile: float,
    driver: bool,
    deviant: bool,
    terms: tuple[str, ...],
    age: float | None = None,
) -> str:
    """Cached entry point: same region + same participant value -> one API call, reused after."""
    return _default_narrator().narrate(
        plain_name, network, structure, bag, percentile, driver, deviant, terms, age
    )


def demo() -> None:
    """Self-check: user message includes the key facts a narrator needs (no live API call)."""
    msg = _build_user_message(
        plain_name="Left Visual (LH_Vis_1)",
        network="Vis",
        structure="cortex",
        bag=1.23,
        percentile=97.0,
        driver=True,
        deviant=True,
        terms=("visual", "semantic", "memory"),
        age=45.0,
    )
    assert "1.23" in msg
    assert "97.0" in msg
    assert "visual, semantic, memory" in msg
    assert "top contributors" in msg
    assert "extreme" in msg
    logger.info("Self-check OK: %s", msg.replace("\n", " | "))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demo()
