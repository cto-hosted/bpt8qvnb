from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

OutputFormat = Literal["wav", "flac", "mp3"]

_FORMAT_SUBTYPE: dict[str, str] = {
    "wav": "PCM_16",
    "flac": "PCM_16",
}


class AudioProcessor:
    def __init__(self, sample_rate: int = 44100, channels: int = 2) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def load(self, path: str | Path) -> tuple[np.ndarray, int]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, sr = sf.read(str(path), always_2d=True)
        if audio.shape[1] != self.channels:
            audio = self._convert_channels(audio, self.channels)
        logger.debug("Loaded %s: shape=%s sr=%d", path.name, audio.shape, sr)
        return audio, sr

    def save(
        self,
        audio: np.ndarray,
        path: str | Path,
        sample_rate: int | None = None,
        output_format: OutputFormat = "wav",
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = sample_rate or self.sample_rate

        if output_format == "mp3":
            wav_path = path.with_suffix(".wav")
            sf.write(str(wav_path), audio, sr, subtype="PCM_16")
            mp3_path = path.with_suffix(".mp3")
            ret = os.system(f'ffmpeg -y -i "{wav_path}" -codec:a libmp3lame -qscale:a 2 "{mp3_path}" 2>/dev/null')
            if ret != 0:
                logger.warning("ffmpeg not available, falling back to WAV")
                return wav_path
            wav_path.unlink(missing_ok=True)
            return mp3_path

        subtype = _FORMAT_SUBTYPE.get(output_format, "PCM_16")
        sf.write(str(path), audio, sr, subtype=subtype)
        logger.debug("Saved %s (format=%s)", path.name, output_format)
        return path

    def generate_silence(self, duration_seconds: float) -> np.ndarray:
        n_samples = int(duration_seconds * self.sample_rate)
        return np.zeros((n_samples, self.channels), dtype=np.float64)

    def normalize(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio
        return audio * (target_peak / peak)

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        return resample_poly(audio, up, down, axis=0)

    def trim_silence(self, audio: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
        mono = np.mean(np.abs(audio), axis=1)
        nonzero = np.where(mono > threshold)[0]
        if len(nonzero) == 0:
            return audio
        return audio[nonzero[0]:nonzero[-1] + 1]

    @staticmethod
    def _convert_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
        src_channels = audio.shape[1]
        if src_channels == 1 and target_channels == 2:
            return np.repeat(audio, 2, axis=1)
        if src_channels == 2 and target_channels == 1:
            return np.mean(audio, axis=1, keepdims=True)
        return audio[:, :target_channels]

    def bars_to_samples(self, bars: int, bpm: float, beats_per_bar: int = 4) -> int:
        beat_duration = 60.0 / bpm
        bar_duration = beat_duration * beats_per_bar
        return int(bars * bar_duration * self.sample_rate)

    def ms_to_samples(self, ms: float) -> int:
        return int(ms * self.sample_rate / 1000.0)
