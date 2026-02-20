from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfilt

from config.settings import RhythmicSettings
from python.pipeline.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class RhythmicPipeline:
    def __init__(self, processor: AudioProcessor, settings: RhythmicSettings) -> None:
        self._processor = processor
        self._settings = settings

    def process(self, audio: np.ndarray) -> np.ndarray:
        logger.info("Starting rhythmic pipeline (bpm=%.1f, bars=%d)", self._settings.bpm, self._settings.bars)
        audio = self._apply_transient_shaping(audio)
        audio = self._apply_rhythmic_eq(audio)
        audio = self._apply_swing(audio)
        audio = self._trim_to_loop(audio)
        audio = self._processor.normalize(audio)
        logger.info("Rhythmic pipeline complete, output shape: %s", audio.shape)
        return audio

    def detect_beats(self, audio: np.ndarray) -> np.ndarray:
        mono = np.mean(audio, axis=1)
        onset_env = self._compute_onset_envelope(mono)
        beat_indices = self._pick_peaks(onset_env, min_distance=self._processor.ms_to_samples(200.0))
        logger.debug("Detected %d beats", len(beat_indices))
        return beat_indices

    def extract_loop(self, audio: np.ndarray) -> np.ndarray:
        target_samples = self._processor.bars_to_samples(
            self._settings.bars,
            self._settings.bpm,
            self._settings.beats_per_bar,
        )
        if len(audio) <= target_samples:
            return audio
        beat_indices = self.detect_beats(audio)
        if len(beat_indices) < 2:
            return audio[:target_samples]
        start = beat_indices[0]
        end = start + target_samples
        if end > len(audio):
            start = 0
            end = target_samples
        return audio[start:end]

    def _apply_transient_shaping(self, audio: np.ndarray) -> np.ndarray:
        threshold = self._settings.transient_threshold
        mono = np.mean(np.abs(audio), axis=1)
        envelope = self._smooth_envelope(mono, window_ms=5.0)
        transient_mask = (mono > threshold).astype(np.float64)
        attack_samples = self._processor.ms_to_samples(2.0)
        shaped = audio.copy()
        for i in range(len(shaped)):
            if transient_mask[i] > 0.5:
                fade_start = max(0, i - attack_samples)
                fade = np.linspace(0.0, 1.0, i - fade_start + 1)
                shaped[fade_start:i + 1] *= fade[:, np.newaxis]
        return shaped

    def _apply_rhythmic_eq(self, audio: np.ndarray) -> np.ndarray:
        sr = self._processor.sample_rate
        sub_sos = butter(2, 80.0 / (sr / 2), btype="low", output="sos")
        mid_sos = butter(2, [200.0 / (sr / 2), 4000.0 / (sr / 2)], btype="band", output="sos")
        sub_audio = sosfilt(sub_sos, audio, axis=0)
        mid_audio = sosfilt(mid_sos, audio, axis=0)
        high_audio = audio - sosfilt(butter(2, 4000.0 / (sr / 2), btype="low", output="sos"), audio, axis=0)
        return sub_audio * 1.1 + mid_audio * 1.0 + high_audio * 0.9

    def _apply_swing(self, audio: np.ndarray) -> np.ndarray:
        if self._settings.swing_factor < 0.01:
            return audio
        beat_samples = int(60.0 / self._settings.bpm * self._processor.sample_rate)
        subdivision = beat_samples // 2
        swing_shift = int(subdivision * self._settings.swing_factor * 0.33)
        if swing_shift == 0:
            return audio
        result = audio.copy()
        i = subdivision
        while i + swing_shift < len(audio) - swing_shift:
            chunk_end = min(i + subdivision, len(audio))
            chunk = audio[i:chunk_end].copy()
            target_start = i + swing_shift
            target_end = min(target_start + len(chunk), len(audio))
            result[target_start:target_end] = chunk[:target_end - target_start]
            i += beat_samples
        return result

    def _trim_to_loop(self, audio: np.ndarray) -> np.ndarray:
        if not self._settings.beat_quantize:
            return audio
        target_samples = self._processor.bars_to_samples(
            self._settings.bars,
            self._settings.bpm,
            self._settings.beats_per_bar,
        )
        if len(audio) >= target_samples:
            return audio[:target_samples]
        pad = np.zeros((target_samples - len(audio), audio.shape[1]), dtype=audio.dtype)
        return np.concatenate([audio, pad], axis=0)

    def _compute_onset_envelope(self, mono: np.ndarray) -> np.ndarray:
        hop = self._processor.ms_to_samples(10.0)
        hop = max(hop, 1)
        n_frames = len(mono) // hop
        envelope = np.zeros(n_frames)
        for i in range(n_frames):
            frame = mono[i * hop:(i + 1) * hop]
            envelope[i] = np.mean(np.abs(frame))
        diff = np.maximum(0.0, np.diff(envelope, prepend=envelope[0]))
        return diff

    def _smooth_envelope(self, signal: np.ndarray, window_ms: float) -> np.ndarray:
        window = max(1, self._processor.ms_to_samples(window_ms))
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode="same")

    @staticmethod
    def _pick_peaks(signal: np.ndarray, min_distance: int = 100) -> np.ndarray:
        peaks: list[int] = []
        last = -min_distance
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
                if i - last >= min_distance:
                    peaks.append(i)
                    last = i
        return np.array(peaks, dtype=np.int64)
