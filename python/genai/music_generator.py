from __future__ import annotations

import logging
from typing import Literal

from python.genai.client import GenAIClient
from python.genai.prompts import PromptContext, build_prompt, build_variation_prompt

logger = logging.getLogger(__name__)

PipelineMode = Literal["rhythmic", "texture"]


class MusicGenerator:
    def __init__(self, client: GenAIClient, temperature: float = 0.9) -> None:
        self._client = client
        self._temperature = temperature

    def generate_spec(
        self,
        description: str,
        mode: PipelineMode,
        bpm: float | None = None,
        key: str | None = None,
        genre: str | None = None,
        duration_bars: int | None = None,
    ) -> str:
        ctx = PromptContext(
            mode=mode,
            description=description,
            bpm=bpm,
            key=key,
            genre=genre,
            duration_bars=duration_bars,
        )
        prompt = build_prompt(ctx)
        logger.info("Generating %s spec for: %s", mode, description)
        spec = self._client.generate_with_config(prompt, temperature=self._temperature)
        logger.info("Spec generated (%d chars)", len(spec))
        return spec

    def generate_variations(
        self,
        original_spec: str,
        count: int,
        mode: PipelineMode,
    ) -> list[str]:
        variations: list[str] = []
        for i in range(1, count + 1):
            logger.info("Generating variation %d/%d", i, count)
            prompt = build_variation_prompt(original_spec, i, mode)
            variation = self._client.generate_with_config(prompt, temperature=self._temperature + 0.05)
            variations.append(variation)
        return variations

    def generate_pack(
        self,
        description: str,
        mode: PipelineMode,
        variation_count: int = 3,
        bpm: float | None = None,
        key: str | None = None,
        genre: str | None = None,
        duration_bars: int | None = None,
    ) -> list[str]:
        base_spec = self.generate_spec(
            description=description,
            mode=mode,
            bpm=bpm,
            key=key,
            genre=genre,
            duration_bars=duration_bars,
        )
        all_specs = [base_spec]
        if variation_count > 0:
            variations = self.generate_variations(base_spec, variation_count, mode)
            all_specs.extend(variations)
        return all_specs
