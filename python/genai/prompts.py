from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PipelineMode = Literal["rhythmic", "texture"]


@dataclass
class PromptContext:
    mode: PipelineMode
    description: str
    bpm: float | None = None
    key: str | None = None
    genre: str | None = None
    duration_bars: int | None = None
    extra: dict | None = None


def build_rhythmic_prompt(ctx: PromptContext) -> str:
    bpm_str = f"{ctx.bpm:.0f} BPM" if ctx.bpm else "a suitable BPM"
    bars_str = f"{ctx.duration_bars} bars" if ctx.duration_bars else "4 bars"
    key_str = f"in the key of {ctx.key}" if ctx.key else ""
    genre_str = f"in a {ctx.genre} style" if ctx.genre else ""

    return f"""You are an expert music producer specializing in rhythmic loop design.
Generate a detailed specification for a rhythmic audio loop with the following requirements:

Description: {ctx.description}
Tempo: {bpm_str}
Length: {bars_str}
{f"Key: {key_str}" if key_str else ""}
{f"Genre: {genre_str}" if genre_str else ""}

Provide a structured specification including:
1. Rhythmic pattern breakdown (kick, snare, hi-hat placement)
2. Transient characteristics (attack sharpness, decay profile)
3. Frequency emphasis (sub-bass, mid presence, high-end sparkle)
4. Groove and swing characteristics
5. Suggested processing chain (compression, EQ, saturation)
6. Loop point recommendations for seamless looping

Keep the specification concise, technical, and actionable for DSP processing."""


def build_texture_prompt(ctx: PromptContext) -> str:
    key_str = f"in the key of {ctx.key}" if ctx.key else ""
    genre_str = f"in a {ctx.genre} style" if ctx.genre else ""

    return f"""You are an expert sound designer specializing in atmospheric texture design.
Generate a detailed specification for a textural audio loop with the following requirements:

Description: {ctx.description}
{f"Key: {key_str}" if key_str else ""}
{f"Genre: {genre_str}" if genre_str else ""}

Provide a structured specification including:
1. Spectral content description (frequency layers, harmonic density)
2. Spatial characteristics (stereo width, depth, reverb character)
3. Temporal evolution (how the texture changes over time)
4. Layering strategy (number of layers, their roles)
5. Modulation and movement (LFO targets, random variation)
6. Processing chain (reverb type, chorus, filtering)
7. Loop point recommendations for seamless, evolving looping

Keep the specification concise, technical, and actionable for DSP processing."""


def build_prompt(ctx: PromptContext) -> str:
    if ctx.mode == "rhythmic":
        return build_rhythmic_prompt(ctx)
    return build_texture_prompt(ctx)


def build_variation_prompt(original_spec: str, variation_index: int, mode: PipelineMode) -> str:
    return f"""Based on the following {'rhythmic loop' if mode == 'rhythmic' else 'texture loop'} specification:

{original_spec}

Generate variation #{variation_index} that:
- Maintains the core character and energy of the original
- Introduces subtle but interesting differences in {'rhythm and groove' if mode == 'rhythmic' else 'spectral texture and evolution'}
- Ensures the variation complements rather than conflicts with the original
- Is suitable for use alongside the original in a sample pack

Provide the same structured specification format as above."""
