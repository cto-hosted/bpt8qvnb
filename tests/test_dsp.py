"""Tests for the C++ DSP engine integration."""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


DSP_BINARY = Path(__file__).parent.parent / "dsp" / "build" / "dsp_engine"


def _write_pcm(path: str, audio: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(audio.astype(np.float32).tobytes())


def _read_pcm(path: str, channels: int = 2) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    data = np.frombuffer(raw, dtype=np.float32)
    if channels > 1 and len(data) % channels == 0:
        return data.reshape(-1, channels)
    return data


def _make_sine_pcm(sr: int = 44100, freq: float = 440.0, duration: float = 2.0, channels: int = 2) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration))
    mono = 0.3 * np.sin(2.0 * np.pi * freq * t)
    if channels == 2:
        return np.stack([mono, mono], axis=1).astype(np.float32)
    return mono[:, np.newaxis].astype(np.float32)


@unittest.skipUnless(DSP_BINARY.exists(), "DSP binary not found - run 'make' in dsp/ first")
class TestDspBinary(unittest.TestCase):
    def test_binary_exists(self) -> None:
        self.assertTrue(DSP_BINARY.exists())
        self.assertTrue(os.access(str(DSP_BINARY), os.X_OK))

    def test_unknown_command_exits_nonzero(self) -> None:
        result = subprocess.run(
            [str(DSP_BINARY), "unknown_command"],
            capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)

    def test_process_command_basic(self) -> None:
        audio = _make_sine_pcm(duration=4.0)
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_in:
            _write_pcm(f_in.name, audio)
            in_path = f_in.name
        out_path = in_path + "_out.pcm"
        try:
            result = subprocess.run(
                [str(DSP_BINARY), "process",
                 "--input", in_path,
                 "--output", out_path,
                 "--sample-rate", "44100",
                 "--channels", "2",
                 "--bpm", "120.0",
                 "--bars", "4"],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")
            self.assertIn("success=true", result.stdout)
            self.assertTrue(os.path.exists(out_path))
            out_audio = _read_pcm(out_path)
            self.assertGreater(len(out_audio), 0)
        finally:
            os.unlink(in_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_crossfade_command(self) -> None:
        audio_a = _make_sine_pcm(freq=440.0, duration=4.0)
        audio_b = _make_sine_pcm(freq=880.0, duration=4.0)
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_a:
            _write_pcm(f_a.name, audio_a)
            path_a = f_a.name
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_b:
            _write_pcm(f_b.name, audio_b)
            path_b = f_b.name
        out_path = path_a + "_crossfade.pcm"
        try:
            result = subprocess.run(
                [str(DSP_BINARY), "crossfade",
                 "--input-a", path_a,
                 "--input-b", path_b,
                 "--output", out_path,
                 "--crossfade-ms", "50.0"],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")
            self.assertIn("samples=", result.stdout)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_similarity_command(self) -> None:
        audio_a = _make_sine_pcm(freq=440.0, duration=2.0)
        audio_b = _make_sine_pcm(freq=440.0, duration=2.0)
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_a:
            _write_pcm(f_a.name, audio_a)
            path_a = f_a.name
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_b:
            _write_pcm(f_b.name, audio_b)
            path_b = f_b.name
        try:
            result = subprocess.run(
                [str(DSP_BINARY), "similarity",
                 "--input-a", path_a,
                 "--input-b", path_b],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")
            self.assertIn("score=", result.stdout)
            score_str = result.stdout.split("score=")[1].split()[0]
            score = float(score_str)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_identical_audio_similarity_is_high(self) -> None:
        audio = _make_sine_pcm(freq=440.0, duration=2.0)
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_a:
            _write_pcm(f_a.name, audio)
            path_a = f_a.name
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_b:
            _write_pcm(f_b.name, audio)
            path_b = f_b.name
        try:
            result = subprocess.run(
                [str(DSP_BINARY), "similarity",
                 "--input-a", path_a,
                 "--input-b", path_b],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0)
            score_str = result.stdout.split("score=")[1].split()[0]
            score = float(score_str)
            self.assertGreater(score, 0.9)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_process_missing_input_fails(self) -> None:
        result = subprocess.run(
            [str(DSP_BINARY), "process",
             "--input", "/nonexistent/file.pcm",
             "--output", "/tmp/out.pcm"],
            capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)

    def test_crossfade_output_written(self) -> None:
        audio_a = _make_sine_pcm(duration=4.0)
        audio_b = _make_sine_pcm(freq=660.0, duration=4.0)
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_a:
            _write_pcm(f_a.name, audio_a)
            path_a = f_a.name
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f_b:
            _write_pcm(f_b.name, audio_b)
            path_b = f_b.name
        out_path = path_a + "_cf_out.pcm"
        try:
            subprocess.run(
                [str(DSP_BINARY), "crossfade",
                 "--input-a", path_a,
                 "--input-b", path_b,
                 "--output", out_path],
                capture_output=True, check=True
            )
            out_size = os.path.getsize(out_path)
            self.assertGreater(out_size, 0)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestDspBinaryNotBuilt(unittest.TestCase):
    @unittest.skipIf(DSP_BINARY.exists(), "DSP binary exists")
    def test_missing_binary_message(self) -> None:
        self.assertFalse(DSP_BINARY.exists())


if __name__ == "__main__":
    unittest.main()
