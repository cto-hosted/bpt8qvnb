from __future__ import annotations

import logging

import numpy as np
from scipy.signal import stft

logger = logging.getLogger(__name__)

_N_MFCC = 13
_N_FFT = 2048
_HOP_LENGTH_MS = 10.0


class SimilarityDetector:
    def __init__(self, sample_rate: int = 44100, threshold: float = 0.85) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._hop_length = max(1, int(_HOP_LENGTH_MS * sample_rate / 1000.0))

    def compute_features(self, audio: np.ndarray) -> np.ndarray:
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        mfccs = self._compute_mfcc(mono)
        spectral_centroid = self._compute_spectral_centroid(mono)
        spectral_rolloff = self._compute_spectral_rolloff(mono)
        rms = self._compute_rms(mono)
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            [np.mean(spectral_centroid)],
            [np.std(spectral_centroid)],
            [np.mean(spectral_rolloff)],
            [np.mean(rms)],
        ])
        return features

    def cosine_similarity(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(features_a)
        norm_b = np.linalg.norm(features_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(features_a, features_b) / (norm_a * norm_b))

    def are_similar(self, audio_a: np.ndarray, audio_b: np.ndarray) -> bool:
        features_a = self.compute_features(audio_a)
        features_b = self.compute_features(audio_b)
        similarity = self.cosine_similarity(features_a, features_b)
        logger.debug("Similarity score: %.4f (threshold=%.2f)", similarity, self.threshold)
        return similarity >= self.threshold

    def find_duplicates(self, audio_clips: list[np.ndarray]) -> list[tuple[int, int, float]]:
        features_list = [self.compute_features(clip) for clip in audio_clips]
        duplicates: list[tuple[int, int, float]] = []
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                score = self.cosine_similarity(features_list[i], features_list[j])
                if score >= self.threshold:
                    duplicates.append((i, j, score))
                    logger.info("Clips %d and %d are similar (score=%.4f)", i, j, score)
        return duplicates

    def rank_by_similarity(
        self, query: np.ndarray, candidates: list[np.ndarray]
    ) -> list[tuple[int, float]]:
        query_features = self.compute_features(query)
        scores = [
            (i, self.cosine_similarity(query_features, self.compute_features(c)))
            for i, c in enumerate(candidates)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _compute_mfcc(self, mono: np.ndarray) -> np.ndarray:
        _, _, spec = stft(mono, fs=self.sample_rate, nperseg=_N_FFT, noverlap=_N_FFT - self._hop_length)
        power_spec = np.abs(spec) ** 2
        n_mels = 40
        mel_filters = self._mel_filterbank(n_mels, _N_FFT // 2 + 1)
        mel_spec = np.dot(mel_filters, power_spec)
        log_mel = np.log(mel_spec + 1e-10)
        mfccs = self._dct(log_mel)[:_N_MFCC]
        return mfccs

    def _mel_filterbank(self, n_mels: int, n_freqs: int) -> np.ndarray:
        fmin = 0.0
        fmax = self.sample_rate / 2.0
        mel_min = self._hz_to_mel(fmin)
        mel_max = self._hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_freqs * 2 / self.sample_rate) * hz_points).astype(int)
        filters = np.zeros((n_mels, n_freqs))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m - f_m_minus > 0:
                    filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, min(f_m_plus, n_freqs)):
                if f_m_plus - f_m > 0:
                    filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return filters

    def _compute_spectral_centroid(self, mono: np.ndarray) -> np.ndarray:
        _, _, spec = stft(mono, fs=self.sample_rate, nperseg=_N_FFT, noverlap=_N_FFT - self._hop_length)
        magnitudes = np.abs(spec)
        freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / self.sample_rate)
        total = np.sum(magnitudes, axis=0)
        total = np.where(total < 1e-10, 1.0, total)
        centroid = np.dot(freqs, magnitudes) / total
        return centroid

    def _compute_spectral_rolloff(self, mono: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
        _, _, spec = stft(mono, fs=self.sample_rate, nperseg=_N_FFT, noverlap=_N_FFT - self._hop_length)
        magnitudes = np.abs(spec)
        cumulative = np.cumsum(magnitudes, axis=0)
        total = cumulative[-1, :]
        threshold = roll_percent * total
        rolloff_idx = np.argmax(cumulative >= threshold[np.newaxis, :], axis=0)
        freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / self.sample_rate)
        return freqs[rolloff_idx]

    def _compute_rms(self, mono: np.ndarray) -> np.ndarray:
        n_frames = len(mono) // self._hop_length
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            frame = mono[i * self._hop_length:(i + 1) * self._hop_length]
            rms[i] = np.sqrt(np.mean(frame ** 2))
        return rms

    @staticmethod
    def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @staticmethod
    def _dct(x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        result = np.zeros_like(x)
        ns = np.arange(n)
        for k in range(n):
            result[k] = np.sum(x * np.cos(np.pi * k * (2 * ns + 1) / (2 * n))[:, np.newaxis], axis=0)
        return result
