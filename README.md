# AI Sample Pack Generator

An AI-powered sample pack generator that combines Google Gemini generative AI with a high-performance C++ DSP engine to produce professional-quality audio loops and sample packs.

## Overview

This tool uses Google Gemini to intelligently design audio specifications, then processes them through a Python audio pipeline backed by a native C++ DSP engine. It supports two creative modes — **rhythmic** (beat-locked loops) and **texture** (ambient/evolving soundscapes) — with similarity detection to ensure pack variety and seamless crossfading for master renders.

## Features

- 🤖 **Google Gemini integration** — generates creative audio specs and variations from plain-text descriptions
- 🎵 **Dual pipeline modes** — rhythmic (BPM-locked) and texture (ambient/layered) processing
- 🔬 **C++ DSP engine** — native loop extraction, crossfade processing, and similarity detection
- 🎚️ **Audio pipeline** — transient shaping, rhythmic EQ, swing quantization, reverb layering
- 🔁 **Similarity deduplication** — cosine-similarity based duplicate detection keeps packs diverse
- 🔗 **Seamless crossfading** — linear, equal-power, and logarithmic fade curves
- 🎛️ **Full CLI** — generate, process, crossfade, and compare audio files from the command line

## Project Structure

```
.
├── main.py                        # Entry point
├── cli/
│   ├── main.py                    # CLI argument parsing
│   └── commands.py                # CLI command implementations
├── python/
│   ├── genai/
│   │   ├── client.py              # Google GenAI API client
│   │   ├── music_generator.py     # Spec generation and variation engine
│   │   └── prompts.py             # Prompt templates and context builders
│   └── pipeline/
│       ├── audio_processor.py     # Core audio I/O, resampling, normalization
│       ├── rhythmic_pipeline.py   # Beat detection, swing, transient shaping
│       ├── texture_pipeline.py    # Layering, reverb, spectral processing
│       ├── crossfade.py           # Crossfade algorithms
│       └── similarity.py         # Feature extraction and cosine similarity
├── dsp/
│   ├── include/
│   │   ├── dsp_engine.hpp
│   │   ├── loop_extractor.hpp
│   │   ├── crossfade_processor.hpp
│   │   └── similarity_detector.hpp
│   ├── src/
│   │   ├── main.cpp
│   │   ├── dsp_engine.cpp
│   │   ├── loop_extractor.cpp
│   │   ├── crossfade_processor.cpp
│   │   └── similarity_detector.cpp
│   └── Makefile
├── config/
│   └── settings.py                # Typed settings dataclasses
├── tests/
│   ├── test_dsp.py
│   ├── test_genai.py
│   └── test_pipeline.py
├── pyproject.toml
├── setup.py
└── requirements.txt
```

## Requirements

- Python 3.10+
- A C++ compiler supporting C++17 (for the DSP engine)
- A [Google AI API key](https://aistudio.google.com/apikey)

## Installation

```bash
# Clone the repository
git clone https://github.com/cto-hosted/bpt8qvnb.git
cd bpt8qvnb

# Install Python dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .

# Build the C++ DSP engine (optional, required for dsp-process command)
cd dsp && make && cd ..
```

## Configuration

Set your Google AI API key as an environment variable:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Generate a Sample Pack

Generate AI-designed loops from a text description:

```bash
# Generate 3 rhythmic loops at 128 BPM in Am
loop-generator generate "dark techno kick with driving bassline" \
  --mode rhythmic \
  --bpm 128 \
  --key Am \
  --genre techno \
  --variations 3 \
  --output-dir output/techno_pack

# Generate ambient texture layers
loop-generator generate "ethereal pads with slow evolution and reverb" \
  --mode texture \
  --density 0.6 \
  --reverb 0.8 \
  --layers 4 \
  --output-dir output/ambient_pack
```

### Process an Existing Audio File

Run an audio file through the rhythmic or texture pipeline:

```bash
loop-generator process input.wav output.wav --mode rhythmic --bpm 120
loop-generator process input.wav output.wav --mode texture --reverb 0.5
```

### Crossfade Two Audio Files

```bash
loop-generator crossfade loop_a.wav loop_b.wav crossfaded.wav --shape equal_power
```

Supported shapes: `linear`, `equal_power`, `logarithmic`

### Similarity Check

Compute a similarity score between two audio files (0.0–1.0):

```bash
loop-generator similarity loop_a.wav loop_b.wav
```

### C++ DSP Engine Processing

Process audio through the native C++ DSP engine (requires building first):

```bash
loop-generator dsp-process input.wav output.wav --bpm 120 --bars 4
```

## CLI Options

All subcommands share a common set of audio and pipeline options:

| Option | Default | Description |
|---|---|---|
| `--sample-rate HZ` | `44100` | Sample rate in Hz |
| `--channels N` | `2` | Audio channels (1 or 2) |
| `--format EXT` | `wav` | Output format: `wav`, `flac`, `mp3` |
| `--output-dir DIR` | `output` | Output directory |
| `--bpm BPM` | `120.0` | Tempo in BPM |
| `--bars N` | `4` | Number of bars per loop |
| `--beats-per-bar N` | `4` | Time signature numerator |
| `--swing FLOAT` | `0.0` | Swing factor (0.0–1.0) |
| `--density FLOAT` | `0.5` | Texture density (0.0–1.0) |
| `--reverb FLOAT` | `0.4` | Reverb amount (0.0–1.0) |
| `--layers N` | `3` | Texture layer count |
| `--similarity-threshold FLOAT` | `0.85` | Deduplication threshold (0.0–1.0) |
| `--crossfade-ms MS` | `50.0` | Crossfade duration in milliseconds |
| `--verbose` / `-v` | off | Enable debug logging |

## Running Tests

```bash
pytest
```

## Building the C++ DSP Engine

```bash
cd dsp
make
# Binary will be at dsp/build/dsp_engine
```

## Architecture

```
CLI Input
   │
   ▼
Google Gemini API ──► MusicGenerator ──► Audio Specs (text)
                                              │
                                              ▼
                                    Python Audio Pipeline
                                    ┌────────────────────┐
                                    │  AudioProcessor     │
                                    │  RhythmicPipeline   │
                                    │  TexturePipeline    │
                                    │  SimilarityDetector │
                                    │  CrossfadeProcessor │
                                    └────────────────────┘
                                              │
                                    ┌─────────┴──────────┐
                                    │                    │
                                    ▼                    ▼
                              Python output      C++ DSP Engine
                              (WAV/FLAC/MP3)    (raw PCM via subprocess)
```

## License

MIT
