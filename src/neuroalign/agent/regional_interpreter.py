"""Regional BAG profile interpreter: which brain networks age differently?"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neuroalign.agent.config import InterpreterConfig
from neuroalign.agent.result import NetworkStat, RegionalInterpretationResult

if TYPE_CHECKING:
    from neuroalign.embedding.result import EmbeddingResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Friendly network names and one-line descriptions
# ---------------------------------------------------------------------------
_CORTICAL_NETWORK_INFO: dict[str, dict] = {
    "Vis": {
        "display": "Visual Network",
        "role": "processes visual information; located in occipital cortex",
    },
    "SomMot": {
        "display": "Somatomotor Network",
        "role": (
            "handles touch, body sensation, and voluntary movement; "
            "spans primary sensory and motor cortex"
        ),
    },
    "DorsAttn": {
        "display": "Dorsal Attention Network",
        "role": (
            "directs spatial attention and coordinates eye movements; "
            "involves parietal and frontal regions"
        ),
    },
    "SalVentAttn": {
        "display": "Salience / Ventral Attention Network",
        "role": (
            "detects important or unexpected events; regulates autonomic responses; "
            "anchored in insula and anterior cingulate cortex"
        ),
    },
    "Limbic": {
        "display": "Limbic Network",
        "role": (
            "processes emotion, reward, and social context; "
            "includes orbitofrontal cortex and temporal poles"
        ),
    },
    "Cont": {
        "display": "Frontoparietal Control Network",
        "role": (
            "supports cognitive control, working memory, and flexible problem-solving; "
            "connects lateral prefrontal and parietal cortex"
        ),
    },
    "Default": {
        "display": "Default Mode Network",
        "role": (
            "active during rest, self-reflection, memory recall, and imagining the future; "
            "hubs in medial prefrontal and posterior cingulate cortex"
        ),
    },
}

_SUBCORTICAL_GROUP_INFO: dict[str, dict] = {
    "Basal Ganglia": {
        "display": "Basal Ganglia",
        "role": (
            "movement initiation, habit learning, and reward processing; "
            "includes putamen, caudate, globus pallidus, and related nuclei"
        ),
        "labels": {"Pu", "Ca", "NAC", "EXA", "GPe", "GPi", "STH"},
    },
    "Substantia Nigra / VTA": {
        "display": "Substantia Nigra & VTA",
        "role": "major dopamine-producing region; central to motivation, reward, and motor control",
        "labels": {"SNc_PBP_VTA", "RN", "SNr", "VeP"},
    },
    "Thalamus": {
        "display": "Thalamus",
        "role": "sensory relay station and attention gatekeeper; connects cortex to subcortical structures",
        "labels": {
            "Pulvinar", "Anterior", "Medio_Dorsal",
            "Ventral_Latero_Dorsal",
            "Central_Lateral-Lateral_Posterior-Medial_Pulvinar",
            "Ventral_Anterior", "Ventral_Latero_Ventral",
        },
    },
    "Hypothalamus & Misc": {
        "display": "Hypothalamus & Adjacent Nuclei",
        "role": (
            "regulates sleep, appetite, stress hormones, and autonomic functions; "
            "includes mammillary, habenula, and subthalamic nuclei"
        ),
        "labels": {"HTH", "HN", "MN"},
    },
    "Hippocampus / Amygdala": {
        "display": "Hippocampus & Amygdala",
        "role": "memory consolidation (hippocampus) and emotional processing / threat detection (amygdala)",
        "labels": {"Hippocampus", "Amygdala"},
    },
    "Cerebellum": {
        "display": "Cerebellum",
        "role": (
            "coordinates movement timing and precision; "
            "also contributes to cognitive and emotional processing"
        ),
        "labels": {"Cerebellar_Region"},  # prefix match
    },
}

_REGIONAL_SYSTEM_PROMPT = (
    "You are the NeuroAlign regional brain interpreter. You help research participants "
    "understand their personal brain aging profile -- specifically, which brain networks or "
    "structures show signs of aging faster or slower than is typical for their age group.\n\n"
    "Key concept: Brain Age Gap (BAG) is the difference between how old a brain region "
    "APPEARS based on its structure and how old the person actually is. A positive z-score "
    "means a region appears structurally OLDER than average for that age; a negative z-score "
    "means it appears YOUNGER / more preserved. These are statistical patterns in a research "
    "context -- not medical findings.\n\n"
    "Your tone is warm, curious, and scientifically grounded. Write for an educated "
    "non-specialist. Use 'you' when addressing the participant. Do NOT make clinical "
    "diagnoses or medical recommendations. Acknowledge that these are group-level patterns "
    "in a research study.\n\n"
    "Return ONLY a valid JSON object (no markdown fences) with exactly these fields:\n"
    "- 'narrative': string -- 2-3 paragraphs explaining the participant regional pattern. "
    "Start with what is most distinctive. Be specific about what the highlighted networks do "
    "and what the pattern might suggest in a research context.\n"
    "- 'network_highlights': list of objects, one per highlighted network/structure, each with:\n"
    "    - 'network': the network key (as given)\n"
    "    - 'direction': 'older' or 'younger'\n"
    "    - 'description': 1-2 sentences on what this means for this specific network\n"
    "- 'overall_summary': string -- one sentence takeaway for the participant."
)


class RegionalProfileInterpreter:
    """Interpret a participant regional BAG z-score profile using an LLM.

    Computes per-network mean z-scores, identifies the most distinctive
    networks/structures, and generates a participant-facing narrative.

    Parameters
    ----------
    atlas_df : pd.DataFrame
        Atlas lookup table with columns: index (int), label (str),
        network_label (str, NaN for subcortical).
    config : InterpreterConfig | None

    Examples
    --------
    ::

        interpreter = RegionalProfileInterpreter.from_parquet(
            "data/processed/long/anatomical_ct.parquet"
        )
        result = interpreter.interpret(embedding_result, "sub-001", "ses-01")
        print(result.narrative)
    """

    def __init__(
        self,
        atlas_df: pd.DataFrame,
        config: InterpreterConfig | None = None,
    ) -> None:
        self.config = config or InterpreterConfig()
        self._client = None
        self._atlas = (
            atlas_df[["index", "label", "network_label"]]
            .drop_duplicates(subset=["index"])
            .set_index("index")
        )
        logger.info("RegionalProfileInterpreter ready (%d atlas entries).", len(self._atlas))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        config: InterpreterConfig | None = None,
    ) -> RegionalProfileInterpreter:
        """Build from a long-format feature-store parquet file."""
        df = pd.read_parquet(parquet_path, columns=["index", "label", "network_label"])
        return cls(df, config=config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily initialise the Gemini client."""
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
            logger.debug("Gemini client initialised (model=%s).", self.config.model)
        return self._client

    def _map_label_to_subcortical_group(self, label: str) -> str | None:
        """Return the subcortical group key for a label, or None if not found."""
        stripped = (
            label.replace("LH-", "").replace("RH-", "")
                 .replace("LH_", "").replace("RH_", "")
        )
        for group, info in _SUBCORTICAL_GROUP_INFO.items():
            for key in info["labels"]:
                if key == "Cerebellar_Region":
                    if stripped.startswith("Cerebellar_Region"):
                        return group
                elif stripped == key or stripped.startswith(key):
                    return group
        return None

    def _compute_network_stats(
        self,
        z_vector: np.ndarray,
        region_cols: list[str],
    ) -> list[NetworkStat]:
        """Compute mean z-score per cortical network and subcortical group."""
        col_to_z: dict[int, float] = {}
        for col, z in zip(region_cols, z_vector):
            try:
                col_to_z[int(col)] = float(z)
            except (ValueError, TypeError):
                continue

        cortical_buckets: dict[str, list[float]] = {k: [] for k in _CORTICAL_NETWORK_INFO}
        subcortical_buckets: dict[str, list[float]] = {k: [] for k in _SUBCORTICAL_GROUP_INFO}

        for idx, z in col_to_z.items():
            if idx not in self._atlas.index:
                continue
            row = self._atlas.loc[idx]
            net = row["network_label"]
            label = row["label"]

            if pd.notna(net) and net in cortical_buckets:
                cortical_buckets[net].append(z)
            else:
                group = self._map_label_to_subcortical_group(str(label))
                if group:
                    subcortical_buckets[group].append(z)

        stats: list[NetworkStat] = []

        for net, zs in cortical_buckets.items():
            if not zs:
                continue
            stats.append(NetworkStat(
                name=net,
                display_name=_CORTICAL_NETWORK_INFO[net]["display"],
                mean_z=float(np.mean(zs)),
                n_parcels=len(zs),
                is_subcortical=False,
            ))

        for group, zs in subcortical_buckets.items():
            if not zs:
                continue
            stats.append(NetworkStat(
                name=group,
                display_name=_SUBCORTICAL_GROUP_INFO[group]["display"],
                mean_z=float(np.mean(zs)),
                n_parcels=len(zs),
                is_subcortical=True,
            ))

        return sorted(stats, key=lambda s: s.abs_z, reverse=True)

    def _build_prompt(self, stats: list[NetworkStat], top_n: int = 5) -> str:
        """Build the user message for the LLM."""
        top = stats[:top_n]
        lines = [
            "## Participant brain aging profile\n",
            (
                "Z-scores represent how a region appears relative to the typical brain "
                "for the participant age group (positive = appears older, "
                "negative = appears younger/more preserved).\n"
            ),
        ]

        lines.append("### Cortical networks (all)")
        for s in sorted([s for s in stats if not s.is_subcortical], key=lambda x: x.abs_z, reverse=True):
            arrow = "up" if s.mean_z > 0 else "down"
            role = _CORTICAL_NETWORK_INFO[s.name]["role"]
            lines.append(
                f"- {s.display_name} ({s.name}): z = {s.mean_z:+.2f} [{arrow}]"
                f"  [{s.n_parcels} parcels]  -- {role}"
            )

        lines.append("\n### Subcortical structures (all)")
        for s in sorted([s for s in stats if s.is_subcortical], key=lambda x: x.abs_z, reverse=True):
            arrow = "up" if s.mean_z > 0 else "down"
            role = _SUBCORTICAL_GROUP_INFO.get(s.name, {}).get("role", "")
            lines.append(
                f"- {s.display_name}: z = {s.mean_z:+.2f} [{arrow}]"
                f"  [{s.n_parcels} structures]  -- {role}"
            )

        lines.append(f"\n### Most distinctive (top {top_n} by absolute z-score)")
        for s in top:
            direction = "appears older" if s.mean_z > 0 else "appears younger"
            lines.append(f"- {s.display_name}: z = {s.mean_z:+.2f} ({direction})")

        lines.append(
            f"\n## Task\nFocus on the top {top_n} most distinctive regions listed above. "
            "For each, write a specific 1-2 sentence description of what this pattern "
            "might suggest in the context of a research participant. "
            "Then write a warm, engaging 2-3 paragraph narrative for the participant "
            "and a one-sentence overall summary."
        )

        return "\n".join(lines)

    def _parse_response(self, raw: str) -> tuple[str, list[dict], str]:
        """Parse LLM JSON response with graceful fallback."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(text)
            narrative = str(data.get("narrative", ""))
            highlights = data.get("network_highlights", [])
            if not isinstance(highlights, list):
                highlights = []
            overall = str(data.get("overall_summary", ""))
            return narrative, highlights, overall
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse regional response as JSON (%s).", exc)
            return raw, [], ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpret(
        self,
        embedding: EmbeddingResult,
        subject: str,
        session: str,
        top_n: int = 5,
    ) -> RegionalInterpretationResult:
        """Interpret a participant regional BAG z-score profile.

        Parameters
        ----------
        embedding : EmbeddingResult
            Fitted embedding result containing z-scored vectors.
        subject : str
            Subject identifier (must be in embedding.vectors).
        session : str
            Session identifier.
        top_n : int
            Number of top networks/structures to highlight.

        Returns
        -------
        RegionalInterpretationResult
        """
        logger.info("Regional interpretation for %s / %s.", subject, session)

        z_vector = embedding.get_vector(subject, session)
        region_cols = embedding.region_cols

        stats = self._compute_network_stats(z_vector, region_cols)
        logger.debug("Computed stats for %d networks/structures.", len(stats))

        user_message = self._build_prompt(stats, top_n=top_n)
        client = self._get_client()
        logger.info("Calling Gemini API for regional profile (model=%s).", self.config.model)

        from google.genai import types as genai_types  # noqa: PLC0415

        response = client.models.generate_content(
            model=self.config.model,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=_REGIONAL_SYSTEM_PROMPT,
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ),
        )
        raw_response = response.text
        logger.debug("Received regional response (%d chars).", len(raw_response))

        narrative, highlights, overall_summary = self._parse_response(raw_response)

        return RegionalInterpretationResult(
            query_subject=subject,
            query_session=session,
            narrative=narrative,
            network_highlights=highlights,
            overall_summary=overall_summary,
            network_stats=stats,
            raw_response=raw_response,
        )
