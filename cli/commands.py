from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

from config.settings import Settings
from python.genai.client import GenAIClient
from python.genai.music_generator import MusicGenerator
from python.pipeline.audio_processor import AudioProcessor
from python.pipeline.crossfade import CrossfadeProcessor
from python.pipeline.rhythmic_pipeline import RhythmicPipeline
from python.pipeline.similarity import SimilarityDetector
from python.pipeline.texture_pipeline import TexturePipeline

logger = logging.getLogger(__name__)

PipelineMode = Literal["rhythmic", "texture"]


def _make_test_audio(settings: Settings, duration_seconds: float = 4.0) -> np.ndarray:
    sr = settings.sample_rate
    n = int(duration_seconds * sr)
    t = np.linspace(0.0, duration_seconds, n)
    freq = 220.0
    audio = 0.3 * np.sin(2.0 * np.pi * freq * t)
    if settings.channels == 2:
        noise = 0.01 * np.random.randn(n)
        audio = np.stack([audio + noise, audio - noise], axis=1)
    else:
        audio = audio[:, np.newaxis]
    return audio.astype(np.float64)


def cmd_generate(
    description: str,
    mode: PipelineMode,
    output_dir: str,
    settings: Settings,
    variation_count: int = 3,
    bpm: float | None = None,
    key: str | None = None,
    genre: str | None = None,
    duration_bars: int | None = None,
) -> list[Path]:
    settings.validate()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = GenAIClient(api_key=settings.api_key, model_name=settings.model_name)
    generator = MusicGenerator(client)

    logger.info("Generating %d specs via GenAI...", variation_count + 1)
    specs = generator.generate_pack(
        description=description,
        mode=mode,
        variation_count=variation_count,
        bpm=bpm or (settings.rhythmic.bpm if mode == "rhythmic" else None),
        key=key,
        genre=genre,
        duration_bars=duration_bars or (settings.rhythmic.bars if mode == "rhythmic" else None),
    )

    processor = AudioProcessor(sample_rate=settings.sample_rate, channels=settings.channels)
    similarity = SimilarityDetector(sample_rate=settings.sample_rate, threshold=settings.similarity_threshold)
    crossfader = CrossfadeProcessor(sample_rate=settings.sample_rate, duration_ms=settings.crossfade_duration_ms)

    if mode == "rhythmic":
        pipeline = RhythmicPipeline(processor, settings.rhythmic)
    else:
        pipeline = TexturePipeline(processor, settings.texture)

    output_paths: list[Path] = []
    processed_clips: list[np.ndarray] = []

    for i, spec in enumerate(specs):
        logger.info("Processing spec %d/%d...", i + 1, len(specs))
        spec_path = out_dir / f"spec_{i:03d}.txt"
        spec_path.write_text(spec)

        audio = _make_test_audio(settings)
        processed = pipeline.process(audio)

        if processed_clips:
            dupes = similarity.find_duplicates(processed_clips + [processed])
            if any(d[1] == len(processed_clips) for d in dupes):
                logger.warning("Clip %d is too similar to an existing clip, skipping.", i)
                continue

        processed_clips.append(processed)
        name = f"loop_{mode}_{i:03d}.{settings.output_format}"
        path = processor.save(processed, out_dir / name, output_format=settings.output_format)
        output_paths.append(path)
        logger.info("Saved: %s", path)

    if len(output_paths) >= 2:
        logger.info("Creating crossfaded master from %d clips...", len(output_paths))
        joined = crossfader.join_seamless(processed_clips)
        master_name = f"master_{mode}_crossfaded.{settings.output_format}"
        master_path = processor.save(joined, out_dir / master_name, output_format=settings.output_format)
        output_paths.append(master_path)
        logger.info("Master saved: %s", master_path)

    return output_paths


def cmd_process(
    input_path: str,
    output_path: str,
    mode: PipelineMode,
    settings: Settings,
) -> Path:
    processor = AudioProcessor(sample_rate=settings.sample_rate, channels=settings.channels)
    audio, sr = processor.load(input_path)
    if sr != settings.sample_rate:
        audio = processor.resample(audio, sr, settings.sample_rate)

    if mode == "rhythmic":
        pipeline = RhythmicPipeline(processor, settings.rhythmic)
    else:
        pipeline = TexturePipeline(processor, settings.texture)

    processed = pipeline.process(audio)
    out_path = Path(output_path)
    saved = processor.save(processed, out_path, output_format=settings.output_format)
    logger.info("Processed and saved to: %s", saved)
    return saved


def cmd_crossfade(
    input_a: str,
    input_b: str,
    output_path: str,
    settings: Settings,
    shape: str = "equal_power",
) -> Path:
    processor = AudioProcessor(sample_rate=settings.sample_rate, channels=settings.channels)
    audio_a, sr_a = processor.load(input_a)
    audio_b, sr_b = processor.load(input_b)
    if sr_a != settings.sample_rate:
        audio_a = processor.resample(audio_a, sr_a, settings.sample_rate)
    if sr_b != settings.sample_rate:
        audio_b = processor.resample(audio_b, sr_b, settings.sample_rate)

    crossfader = CrossfadeProcessor(sample_rate=settings.sample_rate, duration_ms=settings.crossfade_duration_ms)
    result = crossfader.crossfade(audio_a, audio_b, shape=shape)  # type: ignore[arg-type]

    out_path = Path(output_path)
    saved = processor.save(result, out_path, output_format=settings.output_format)
    logger.info("Crossfaded and saved to: %s", saved)
    return saved


def cmd_similarity(
    input_a: str,
    input_b: str,
    settings: Settings,
) -> tuple[float, bool]:
    processor = AudioProcessor(sample_rate=settings.sample_rate, channels=settings.channels)
    audio_a, _ = processor.load(input_a)
    audio_b, _ = processor.load(input_b)
    detector = SimilarityDetector(sample_rate=settings.sample_rate, threshold=settings.similarity_threshold)
    features_a = detector.compute_features(audio_a)
    features_b = detector.compute_features(audio_b)
    score = detector.cosine_similarity(features_a, features_b)
    is_similar = score >= settings.similarity_threshold
    logger.info("Similarity score: %.4f (similar=%s)", score, is_similar)
    return score, is_similar


def cmd_dsp_process(
    input_path: str,
    output_path: str,
    settings: Settings,
) -> Path:
    dsp_binary = settings.dsp_binary
    if not os.path.exists(dsp_binary):
        raise FileNotFoundError(
            f"DSP engine binary not found at {dsp_binary}. "
            f"Run 'make' in the dsp/ directory first."
        )
    processor = AudioProcessor(sample_rate=settings.sample_rate, channels=settings.channels)
    audio, sr = processor.load(input_path)
    if sr != settings.sample_rate:
        audio = processor.resample(audio, sr, settings.sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as tmp_in:
        tmp_in.write(audio.astype(np.float32).tobytes())
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path + "_out.pcm"
    try:
        cmd = [
            dsp_binary, "process",
            "--input", tmp_in_path,
            "--output", tmp_out_path,
            "--sample-rate", str(settings.sample_rate),
            "--channels", str(settings.channels),
            "--bpm", str(settings.rhythmic.bpm),
            "--bars", str(settings.rhythmic.bars),
            "--crossfade-ms", str(settings.crossfade_duration_ms),
            "--similarity-thresh", str(settings.similarity_threshold),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug("DSP stdout: %s", result.stdout.strip())

        with open(tmp_out_path, "rb") as f:
            raw = f.read()
        n_floats = len(raw) // 4
        out_audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, settings.channels)
        out_path = Path(output_path)
        saved = processor.save(out_audio, out_path, output_format=settings.output_format)
        return saved
    finally:
        os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)
