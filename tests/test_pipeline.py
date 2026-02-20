"""Tests for the audio processing pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from config.settings import RhythmicSettings, Settings, TextureSettings
from python.pipeline.audio_processor import AudioProcessor
from python.pipeline.crossfade import CrossfadeProcessor
from python.pipeline.rhythmic_pipeline import RhythmicPipeline
from python.pipeline.similarity import SimilarityDetector
from python.pipeline.texture_pipeline import TexturePipeline


def _make_sine(sr: int = 44100, freq: float = 440.0, duration: float = 2.0, channels: int = 2) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration))
    mono = 0.5 * np.sin(2.0 * np.pi * freq * t)
    if channels == 2:
        return np.stack([mono, mono * 0.9], axis=1)
    return mono[:, np.newaxis]


class TestAudioProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.proc = AudioProcessor(sample_rate=44100, channels=2)

    def test_normalize_peak(self) -> None:
        audio = _make_sine()
        normalized = self.proc.normalize(audio, target_peak=0.95)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 0.95, places=3)

    def test_normalize_silence(self) -> None:
        silence = np.zeros((1000, 2))
        result = self.proc.normalize(silence)
        np.testing.assert_array_equal(result, silence)

    def test_generate_silence_shape(self) -> None:
        silence = self.proc.generate_silence(1.0)
        self.assertEqual(silence.shape[0], 44100)
        self.assertEqual(silence.shape[1], 2)

    def test_bars_to_samples(self) -> None:
        samples = self.proc.bars_to_samples(4, bpm=120.0, beats_per_bar=4)
        expected = int(4 * 4 * (60.0 / 120.0) * 44100)
        self.assertEqual(samples, expected)

    def test_ms_to_samples(self) -> None:
        self.assertEqual(self.proc.ms_to_samples(1000.0), 44100)
        self.assertEqual(self.proc.ms_to_samples(0.0), 0)

    def test_resample_same_rate(self) -> None:
        audio = _make_sine()
        result = self.proc.resample(audio, 44100, 44100)
        np.testing.assert_array_equal(result, audio)

    def test_trim_silence(self) -> None:
        audio = np.zeros((200, 2))
        audio[50:150] = 0.5
        trimmed = self.proc.trim_silence(audio, threshold=0.1)
        self.assertLess(len(trimmed), len(audio))

    def test_channel_conversion_mono_to_stereo(self) -> None:
        mono = _make_sine(channels=1)
        stereo = AudioProcessor._convert_channels(mono, 2)
        self.assertEqual(stereo.shape[1], 2)

    def test_channel_conversion_stereo_to_mono(self) -> None:
        stereo = _make_sine(channels=2)
        mono = AudioProcessor._convert_channels(stereo, 1)
        self.assertEqual(mono.shape[1], 1)


class TestRhythmicPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.proc = AudioProcessor(sample_rate=44100, channels=2)
        self.settings = RhythmicSettings(bpm=120.0, bars=2)
        self.pipeline = RhythmicPipeline(self.proc, self.settings)

    def test_process_returns_ndarray(self) -> None:
        audio = _make_sine(duration=4.0)
        result = self.pipeline.process(audio)
        self.assertIsInstance(result, np.ndarray)

    def test_process_output_channels(self) -> None:
        audio = _make_sine(duration=4.0)
        result = self.pipeline.process(audio)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], 2)

    def test_extract_loop_trims_to_bars(self) -> None:
        audio = _make_sine(duration=8.0)
        extracted = self.pipeline.extract_loop(audio)
        target = self.proc.bars_to_samples(2, 120.0, 4)
        self.assertEqual(len(extracted), target)

    def test_detect_beats_returns_array(self) -> None:
        audio = _make_sine(duration=4.0)
        beats = self.pipeline.detect_beats(audio)
        self.assertIsInstance(beats, np.ndarray)

    def test_process_normalizes_output(self) -> None:
        audio = _make_sine(duration=4.0) * 0.01
        result = self.pipeline.process(audio)
        self.assertGreater(float(np.max(np.abs(result))), 0.1)


class TestTexturePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.proc = AudioProcessor(sample_rate=44100, channels=2)
        self.settings = TextureSettings(density=0.5, layer_count=2, reverb_amount=0.3)
        self.pipeline = TexturePipeline(self.proc, self.settings)

    def test_process_returns_ndarray(self) -> None:
        audio = _make_sine(duration=2.0)
        result = self.pipeline.process(audio)
        self.assertIsInstance(result, np.ndarray)

    def test_process_preserves_channels(self) -> None:
        audio = _make_sine(duration=2.0)
        result = self.pipeline.process(audio)
        self.assertEqual(result.shape[1], 2)

    def test_layer_returns_mixed(self) -> None:
        audio = _make_sine(duration=2.0)
        layered = self.pipeline.layer(audio)
        self.assertEqual(layered.shape, audio.shape)

    def test_process_normalizes_output(self) -> None:
        audio = _make_sine(duration=2.0) * 0.001
        result = self.pipeline.process(audio)
        self.assertGreater(float(np.max(np.abs(result))), 0.05)


class TestSimilarityDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = SimilarityDetector(sample_rate=44100, threshold=0.85)

    def test_identical_clips_are_similar(self) -> None:
        audio = _make_sine(duration=1.0)
        self.assertTrue(self.detector.are_similar(audio, audio))

    def test_identical_clips_high_score(self) -> None:
        audio = _make_sine(duration=1.0)
        features = self.detector.compute_features(audio)
        score = self.detector.cosine_similarity(features, features)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_different_clips_lower_score(self) -> None:
        audio_a = _make_sine(freq=220.0, duration=1.0)
        audio_b = _make_sine(freq=880.0, duration=1.0)
        features_a = self.detector.compute_features(audio_a)
        features_b = self.detector.compute_features(audio_b)
        score = self.detector.cosine_similarity(features_a, features_b)
        self.assertLess(score, 1.0)

    def test_find_duplicates_with_identical(self) -> None:
        audio = _make_sine(duration=1.0)
        dupes = self.detector.find_duplicates([audio, audio, _make_sine(freq=880.0)])
        self.assertGreater(len(dupes), 0)
        self.assertTrue(dupes[0][2] >= 0.85)

    def test_rank_by_similarity(self) -> None:
        query = _make_sine(freq=440.0, duration=1.0)
        candidates = [
            _make_sine(freq=440.0, duration=1.0),
            _make_sine(freq=880.0, duration=1.0),
        ]
        ranked = self.detector.rank_by_similarity(query, candidates)
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])

    def test_zero_norm_similarity(self) -> None:
        zero = np.zeros(10)
        score = self.detector.cosine_similarity(zero, zero)
        self.assertEqual(score, 0.0)


class TestCrossfadeProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.proc = CrossfadeProcessor(sample_rate=44100, duration_ms=10.0)

    def test_crossfade_output_length(self) -> None:
        a = _make_sine(duration=1.0)
        b = _make_sine(freq=880.0, duration=1.0)
        result = self.proc.crossfade(a, b)
        n_fade = self.proc.duration_samples
        expected = len(a) + len(b) - n_fade
        self.assertEqual(len(result), expected)

    def test_fade_in_shapes(self) -> None:
        audio = _make_sine(duration=0.5)
        for shape in ("linear", "equal_power", "logarithmic"):
            result = self.proc.fade_in(audio, shape=shape)  # type: ignore[arg-type]
            self.assertEqual(result.shape, audio.shape)

    def test_fade_out_shapes(self) -> None:
        audio = _make_sine(duration=0.5)
        for shape in ("linear", "equal_power", "logarithmic"):
            result = self.proc.fade_out(audio, shape=shape)  # type: ignore[arg-type]
            self.assertEqual(result.shape, audio.shape)

    def test_make_seamless_shape(self) -> None:
        audio = _make_sine(duration=1.0)
        result = self.proc.make_seamless(audio)
        self.assertEqual(result.shape, audio.shape)

    def test_join_seamless_single_clip(self) -> None:
        audio = _make_sine(duration=1.0)
        result = self.proc.join_seamless([audio])
        np.testing.assert_array_equal(result, audio)

    def test_join_seamless_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.proc.join_seamless([])

    def test_fade_in_starts_quiet(self) -> None:
        audio = np.ones((44100, 2))
        result = self.proc.fade_in(audio)
        self.assertAlmostEqual(float(result[0, 0]), 0.0, places=3)

    def test_fade_out_ends_quiet(self) -> None:
        audio = np.ones((44100, 2))
        result = self.proc.fade_out(audio)
        self.assertAlmostEqual(float(result[-1, 0]), 0.0, places=3)


class TestSettings(unittest.TestCase):
    def test_validate_raises_without_api_key(self) -> None:
        from config.settings import Settings
        s = Settings(api_key="")
        with self.assertRaises(ValueError):
            s.validate()

    def test_validate_passes_with_api_key(self) -> None:
        s = Settings(api_key="test-key")
        s.validate()

    def test_validate_raises_bad_sample_rate(self) -> None:
        s = Settings(api_key="key", sample_rate=8000)
        with self.assertRaises(ValueError):
            s.validate()

    def test_validate_raises_bad_format(self) -> None:
        s = Settings(api_key="key", output_format="ogg")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            s.validate()


if __name__ == "__main__":
    unittest.main()
