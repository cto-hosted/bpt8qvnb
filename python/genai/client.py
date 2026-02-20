from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GenAIClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
        if not api_key:
            raise ValueError("API key must not be empty")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        logger.debug("GenAIClient initialised with model %s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_text(self, prompt: str, **generation_kwargs: Any) -> str:
        logger.debug("Sending prompt to GenAI (length=%d)", len(prompt))
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            **generation_kwargs,
        )
        text = response.text.strip()
        logger.debug("Received response (length=%d)", len(text))
        return text

    def generate_with_config(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_output_tokens: int = 2048,
    ) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return self.generate_text(prompt, config=config)
