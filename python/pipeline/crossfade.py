from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

FadeShape = Literal["linear", "equal_power", "logarithmic"]


class CrossfadeProcessor:
    def __init__(self, sample_rate: int = 44100, duration_ms: float = 50.0) -> None:
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms

    @property
    def duration_samples(self) -> int:
        return max(1, int(self.duration_ms * self.sample_rate / 1000.0))

    def crossfade(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        shape: FadeShape = "equal_power",
    ) -> np.ndarray:
        n_fade = min(self.duration_samples, len(audio_a), len(audio_b))
        logger.debug(
            "Crossfading %d samples (shape=%s, fade=%d)",
            len(audio_a) + len(audio_b),
            shape,
            n_fade,
        )
        fade_out, fade_in = self._build_fade_curves(n_fade, shape)
        if audio_a.ndim == 2:
            fade_out = fade_out[:, np.newaxis]
            fade_in = fade_in[:, np.newaxis]
        body_a = audio_a[:-n_fade]
        tail_a = audio_a[-n_fade:]
        head_b = audio_b[:n_fade]
        body_b = audio_b[n_fade:]
        crossfade_region = tail_a * fade_out + head_b * fade_in
        return np.concatenate([body_a, crossfade_region, body_b], axis=0)

    def fade_in(self, audio: np.ndarray, shape: FadeShape = "equal_power") -> np.ndarray:
        n_fade = min(self.duration_samples, len(audio))
        _, fade_curve = self._build_fade_curves(n_fade, shape)
        result = audio.copy()
        if audio.ndim == 2:
            result[:n_fade] *= fade_curve[:, np.newaxis]
        else:
            result[:n_fade] *= fade_curve
        return result

    def fade_out(self, audio: np.ndarray, shape: FadeShape = "equal_power") -> np.ndarray:
        n_fade = min(self.duration_samples, len(audio))
        fade_curve, _ = self._build_fade_curves(n_fade, shape)
        result = audio.copy()
        if audio.ndim == 2:
            result[-n_fade:] *= fade_curve[:, np.newaxis]
        else:
            result[-n_fade:] *= fade_curve
        return result

    def make_seamless(
        self, audio: np.ndarray, shape: FadeShape = "equal_power"
    ) -> np.ndarray:
        n_fade = min(self.duration_samples, len(audio) // 4)
        fade_out_curve, fade_in_curve = self._build_fade_curves(n_fade, shape)
        result = audio.copy()
        if audio.ndim == 2:
            result[:n_fade] *= fade_in_curve[:, np.newaxis]
            result[-n_fade:] *= fade_out_curve[:, np.newaxis]
        else:
            result[:n_fade] *= fade_in_curve
            result[-n_fade:] *= fade_out_curve
        return result

    def join_seamless(
        self,
        clips: list[np.ndarray],
        shape: FadeShape = "equal_power",
    ) -> np.ndarray:
        if not clips:
            raise ValueError("clips list must not be empty")
        if len(clips) == 1:
            return clips[0]
        result = clips[0]
        for clip in clips[1:]:
            result = self.crossfade(result, clip, shape=shape)
        return result

    def _build_fade_curves(
        self, n_samples: int, shape: FadeShape
    ) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 1.0, n_samples)
        if shape == "linear":
            fade_out = 1.0 - t
            fade_in = t
        elif shape == "equal_power":
            fade_out = np.cos(t * np.pi / 2.0)
            fade_in = np.sin(t * np.pi / 2.0)
        elif shape == "logarithmic":
            fade_out = np.sqrt(1.0 - t)
            fade_in = np.sqrt(t)
        else:
            raise ValueError(f"Unknown fade shape: {shape}")
        return fade_out, fade_in
