"""LLM-powered interpreter for brain twins retrieval results (Gemini backend)."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import TYPE_CHECKING

import pandas as pd

from neuroalign.agent.config import InterpreterConfig
from neuroalign.agent.prompts import SYSTEM_PROMPT, build_user_message
from neuroalign.agent.result import InterpretationResult

if TYPE_CHECKING:
    from neuroalign.retrieval.result import RetrievalResult

logger = logging.getLogger(__name__)


class BrainTwinsInterpreter:
    """Interpret brain twins retrieval results using the Gemini API (google-genai SDK)."""

    def __init__(self, config: InterpreterConfig | None = None) -> None:
        self.config = config or InterpreterConfig()
        self._model = None

    def _get_client(self):
        if self._model is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GOOGLE_API_KEY environment variable is not set. "
                    "Get a free key at aistudio.google.com, then add it to .env."
                )
            try:
                from google import genai  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "The 'google-genai' package is required: pip install google-genai"
                ) from exc
            self._model = genai.Client(api_key=api_key)
            logger.debug("Gemini client initialised (model=%s).", self.config.model)
        return self._model

    @staticmethod
    def _has_questionnaire_data(row: dict) -> bool:
        skip_keys = {"rank", "Subject Code", "subject_code", "subject"}
        for key, value in row.items():
            if key in skip_keys:
                continue
            if value is None:
                continue
            try:
                if isinstance(value, float) and math.isnan(value):
                    continue
            except (TypeError, ValueError):
                pass
            return True
        return False

    def _parse_response(self, raw: str) -> tuple[str, list[str], str, list[dict]]:
        """Parse the LLM JSON response.

        Returns (narrative, key_shared_traits, cluster_profile, scale_highlights).
        Falls back gracefully if the model returns non-JSON.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0].strip()
        try:
            data = json.loads(text)
            narrative = str(data.get("narrative", ""))
            traits = data.get("key_shared_traits", [])
            if not isinstance(traits, list):
                traits = []
            traits = [str(t) for t in traits]
            cluster_profile = str(data.get("cluster_profile", ""))
            scale_highlights = data.get("scale_highlights", [])
            if not isinstance(scale_highlights, list):
                scale_highlights = []
            logger.debug("JSON response parsed successfully.")
            return narrative, traits, cluster_profile, scale_highlights
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse LLM response as JSON (%s). Using raw text as narrative.", exc
            )
            return raw, [], "", []

    def interpret(
        self,
        result: RetrievalResult,
        profiles: pd.DataFrame,
        summary: dict,
    ) -> InterpretationResult:
        """Run LLM interpretation for a brain twins retrieval result."""
        logger.info(
            "Interpreting brain twins for %s / %s.",
            result.query_subject,
            result.query_session,
        )

        query_rows = profiles[profiles["rank"] == 0]
        query_row: dict | None = None
        if self.config.include_query_profile and not query_rows.empty:
            query_row = query_rows.iloc[0].to_dict()
        elif self.config.include_query_profile:
            logger.warning("No query profile row found (rank == 0).")

        twin_df = profiles[profiles["rank"] > 0].sort_values("rank")
        twin_rows: list[dict] = [row.to_dict() for _, row in twin_df.iterrows()]

        twins_with_data = [r for r in twin_rows if self._has_questionnaire_data(r)]
        n_with_data = len(twins_with_data)
        logger.debug("%d / %d twins have questionnaire data.", n_with_data, len(twin_rows))

        if n_with_data < self.config.min_twins_with_data:
            raise ValueError(
                f"Only {n_with_data} brain twin(s) have questionnaire data, "
                f"but at least {self.config.min_twins_with_data} are required."
            )

        user_message = build_user_message(
            query_row=query_row,
            twin_rows=twin_rows,
            summary=summary,
            include_cols=self.config.include_cols,
        )
        logger.debug("User message built (%d chars).", len(user_message))

        client = self._get_client()
        logger.info("Calling Gemini API (model=%s).", self.config.model)

        from google.genai import types as genai_types  # noqa: PLC0415

        response = client.models.generate_content(
            model=self.config.model,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ),
        )
        raw_response = response.text
        logger.debug("Received response (%d chars).", len(raw_response))

        narrative, key_shared_traits, cluster_profile, scale_highlights = self._parse_response(
            raw_response
        )

        return InterpretationResult(
            query_subject=result.query_subject,
            query_session=result.query_session,
            narrative=narrative,
            key_shared_traits=key_shared_traits,
            cluster_profile=cluster_profile,
            raw_response=raw_response,
            n_twins_used=n_with_data,
            scale_highlights=scale_highlights,
        )
