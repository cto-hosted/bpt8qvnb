from __future__ import annotations

import argparse
import logging
import sys
from typing import Literal

from config.settings import RhythmicSettings, Settings, TextureSettings


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_settings(args: argparse.Namespace) -> Settings:
    rhythmic = RhythmicSettings(
        bpm=args.bpm,
        bars=args.bars,
        beats_per_bar=args.beats_per_bar,
        swing_factor=args.swing,
    )
    texture = TextureSettings(
        density=args.density,
        reverb_amount=args.reverb,
        layer_count=args.layers,
    )
    return Settings(
        sample_rate=args.sample_rate,
        channels=args.channels,
        output_format=args.format,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        crossfade_duration_ms=args.crossfade_ms,
        rhythmic=rhythmic,
        texture=texture,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=44100, metavar="HZ",
                        help="Sample rate in Hz (default: 44100)")
    parser.add_argument("--channels", type=int, default=2, choices=[1, 2],
                        help="Number of audio channels (default: 2)")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "flac", "mp3"],
                        help="Output audio format (default: wav)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, metavar="FLOAT",
                        help="Similarity threshold 0..1 (default: 0.85)")
    parser.add_argument("--crossfade-ms", type=float, default=50.0, metavar="MS",
                        help="Crossfade duration in milliseconds (default: 50.0)")
    parser.add_argument("--bpm", type=float, default=120.0, metavar="BPM",
                        help="Tempo in BPM for rhythmic mode (default: 120.0)")
    parser.add_argument("--bars", type=int, default=4, metavar="N",
                        help="Number of bars for rhythmic loop (default: 4)")
    parser.add_argument("--beats-per-bar", type=int, default=4, metavar="N",
                        help="Beats per bar (default: 4)")
    parser.add_argument("--swing", type=float, default=0.0, metavar="FLOAT",
                        help="Swing factor 0..1 (default: 0.0)")
    parser.add_argument("--density", type=float, default=0.5, metavar="FLOAT",
                        help="Texture density 0..1 (default: 0.5)")
    parser.add_argument("--reverb", type=float, default=0.4, metavar="FLOAT",
                        help="Reverb amount 0..1 (default: 0.4)")
    parser.add_argument("--layers", type=int, default=3, metavar="N",
                        help="Number of texture layers (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loop-generator",
        description="AI-powered loop generator with Python GenAI and C++ DSP engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate loops using Google GenAI")
    generate.add_argument("description", type=str, help="Text description of the loop to generate")
    generate.add_argument("--mode", type=str, default="rhythmic", choices=["rhythmic", "texture"],
                          help="Pipeline mode: rhythmic or texture (default: rhythmic)")
    generate.add_argument("--variations", type=int, default=3, metavar="N",
                          help="Number of variations to generate (default: 3)")
    generate.add_argument("--key", type=str, default=None, metavar="KEY",
                          help="Musical key (e.g. Am, C, F#m)")
    generate.add_argument("--genre", type=str, default=None, metavar="GENRE",
                          help="Musical genre (e.g. techno, ambient, hip-hop)")
    _add_common_args(generate)

    process = subparsers.add_parser("process", help="Process an existing audio file through the pipeline")
    process.add_argument("input", type=str, help="Input audio file path")
    process.add_argument("output", type=str, help="Output audio file path")
    process.add_argument("--mode", type=str, default="rhythmic", choices=["rhythmic", "texture"],
                         help="Pipeline mode: rhythmic or texture (default: rhythmic)")
    _add_common_args(process)

    crossfade = subparsers.add_parser("crossfade", help="Crossfade two audio files together")
    crossfade.add_argument("input_a", type=str, help="First input audio file")
    crossfade.add_argument("input_b", type=str, help="Second input audio file")
    crossfade.add_argument("output", type=str, help="Output audio file path")
    crossfade.add_argument("--shape", type=str, default="equal_power",
                           choices=["linear", "equal_power", "logarithmic"],
                           help="Crossfade curve shape (default: equal_power)")
    _add_common_args(crossfade)

    similarity = subparsers.add_parser("similarity", help="Compute similarity between two audio files")
    similarity.add_argument("input_a", type=str, help="First input audio file")
    similarity.add_argument("input_b", type=str, help="Second input audio file")
    _add_common_args(similarity)

    dsp = subparsers.add_parser("dsp-process", help="Process audio using the C++ DSP engine binary")
    dsp.add_argument("input", type=str, help="Input audio file path")
    dsp.add_argument("output", type=str, help="Output audio file path")
    _add_common_args(dsp)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    settings = _build_settings(args)

    try:
        if args.command == "generate":
            from cli.commands import cmd_generate
            paths = cmd_generate(
                description=args.description,
                mode=args.mode,
                output_dir=args.output_dir,
                settings=settings,
                variation_count=args.variations,
                key=args.key,
                genre=args.genre,
            )
            print(f"Generated {len(paths)} files:")
            for p in paths:
                print(f"  {p}")

        elif args.command == "process":
            from cli.commands import cmd_process
            path = cmd_process(
                input_path=args.input,
                output_path=args.output,
                mode=args.mode,
                settings=settings,
            )
            print(f"Processed: {path}")

        elif args.command == "crossfade":
            from cli.commands import cmd_crossfade
            path = cmd_crossfade(
                input_a=args.input_a,
                input_b=args.input_b,
                output_path=args.output,
                settings=settings,
                shape=args.shape,
            )
            print(f"Crossfaded: {path}")

        elif args.command == "similarity":
            from cli.commands import cmd_similarity
            score, is_similar = cmd_similarity(
                input_a=args.input_a,
                input_b=args.input_b,
                settings=settings,
            )
            print(f"Similarity score: {score:.4f}")
            print(f"Similar: {is_similar}")

        elif args.command == "dsp-process":
            from cli.commands import cmd_dsp_process
            path = cmd_dsp_process(
                input_path=args.input,
                output_path=args.output,
                settings=settings,
            )
            print(f"DSP processed: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
