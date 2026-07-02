"""Configuration for the BrainTwinsInterpreter."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InterpreterConfig(BaseModel):
    """Configuration for the LLM-powered brain twins interpreter.

    Attributes
    ----------
    model : str
        Gemini model identifier (e.g. "gemini-2.0-flash", "gemini-2.0-pro").
    max_tokens : int
        Maximum tokens in the LLM response.
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    include_cols : list[str] | None
        Questionnaire columns to include. None means use all available
        PROFILE_COLS from QuestionnaireLookup.
    min_twins_with_data : int
        Minimum number of twins with at least one non-null questionnaire
        field required to proceed with interpretation.
    include_query_profile : bool
        Whether to include the query participant's own profile in the prompt.
    """

    model: str = "gemini-2.5-flash"
    max_tokens: int = 15000
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    include_cols: list[str] | None = None
    min_twins_with_data: int = Field(default=2, ge=1)
    include_query_profile: bool = True
