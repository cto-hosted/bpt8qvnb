from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfilt

from config.settings import TextureSettings
from python.pipeline.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class TexturePipeline:
    def __init__(self, processor: AudioProcessor, settings: TextureSettings) -> None:
        self._processor = processor
        self._settings = settings

    def process(self, audio: np.ndarray) -> np.ndarray:
        logger.info(
            "Starting texture pipeline (density=%.2f, layers=%d)",
            self._settings.density,
            self._settings.layer_count,
        )
        audio = self._apply_spectral_shaping(audio)
        audio = self._apply_stereo_widening(audio)
        audio = self._apply_reverb(audio)
        audio = self._apply_temporal_evolution(audio)
        audio = self._processor.normalize(audio)
        logger.info("Texture pipeline complete, output shape: %s", audio.shape)
        return audio

    def layer(self, audio: np.ndarray) -> np.ndarray:
        layers: list[np.ndarray] = [audio]
        for i in range(1, self._settings.layer_count):
            layer = self._create_layer(audio, layer_index=i)
            layers.append(layer)
        mixed = np.mean(np.stack(layers, axis=0), axis=0)
        return mixed

    def _apply_spectral_shaping(self, audio: np.ndarray) -> np.ndarray:
        sr = self._processor.sample_rate
        spread = self._settings.spectral_spread
        low_gain = 0.8 + 0.4 * (1.0 - spread)
        high_gain = 0.6 + 0.8 * spread
        low_sos = butter(2, 500.0 / (sr / 2), btype="low", output="sos")
        high_sos = butter(2, 500.0 / (sr / 2), btype="high", output="sos")
        low_band = sosfilt(low_sos, audio, axis=0) * low_gain
        high_band = sosfilt(high_sos, audio, axis=0) * high_gain
        return low_band + high_band

    def _apply_stereo_widening(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[1] < 2:
            return audio
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5
        width_factor = 0.5 + self._settings.spectral_spread
        side_wide = side * width_factor
        left = mid + side_wide
        right = mid - side_wide
        return np.stack([left, right], axis=1)

    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        amount = self._settings.reverb_amount
        if amount < 0.01:
            return audio
        sr = self._processor.sample_rate
        delay_times_ms = [23.0, 53.0, 97.0, 149.0]
        wet = np.zeros_like(audio)
        for delay_ms in delay_times_ms:
            delay_samples = self._processor.ms_to_samples(delay_ms)
            decay = 0.6 ** (delay_ms / 50.0)
            delayed = np.zeros_like(audio)
            if delay_samples < len(audio):
                delayed[delay_samples:] = audio[:-delay_samples] * decay
            wet += delayed
        wet = wet / len(delay_times_ms)
        decay_sos = butter(1, 4000.0 / (sr / 2), btype="low", output="sos")
        wet = sosfilt(decay_sos, wet, axis=0)
        return audio * (1.0 - amount) + wet * amount

    def _apply_temporal_evolution(self, audio: np.ndarray) -> np.ndarray:
        rate = self._settings.evolution_rate
        if rate < 0.01:
            return audio
        n_samples = len(audio)
        t = np.linspace(0.0, 2.0 * np.pi, n_samples)
        lfo = 1.0 + rate * 0.3 * np.sin(t * 0.5)[:, np.newaxis]
        return audio * lfo

    def _create_layer(self, audio: np.ndarray, layer_index: int) -> np.ndarray:
        sr = self._processor.sample_rate
        detune_cents = layer_index * 7.0
        pitch_ratio = 2.0 ** (detune_cents / 1200.0)
        from scipy.signal import resample_poly
        from math import gcd
        num = round(pitch_ratio * 1000)
        den = 1000
        g = gcd(num, den)
        num, den = num // g, den // g
        resampled = resample_poly(audio, den, num, axis=0)
        if len(resampled) < len(audio):
            pad = np.zeros((len(audio) - len(resampled), audio.shape[1]), dtype=audio.dtype)
            resampled = np.concatenate([resampled, pad], axis=0)
        else:
            resampled = resampled[:len(audio)]
        delay_samples = self._processor.ms_to_samples(layer_index * 13.0)
        delayed = np.zeros_like(audio)
        if delay_samples < len(audio):
            delayed[delay_samples:] = resampled[:-delay_samples] if delay_samples > 0 else resampled
        else:
            delayed = resampled
        amplitude = 0.7 ** layer_index
        return delayed * amplitude
