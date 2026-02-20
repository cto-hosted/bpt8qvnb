import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RhythmicSettings:
    bpm: float = 120.0
    beats_per_bar: int = 4
    bars: int = 4
    transient_threshold: float = 0.3
    beat_quantize: bool = True
    swing_factor: float = 0.0


@dataclass
class TextureSettings:
    density: float = 0.5
    spectral_spread: float = 0.7
    reverb_amount: float = 0.4
    layer_count: int = 3
    evolution_rate: float = 0.2


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    model_name: str = "gemini-1.5-flash"
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    output_format: Literal["wav", "flac", "mp3"] = "wav"
    output_dir: str = "output"
    dsp_binary: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dsp", "build", "dsp_engine")
    similarity_threshold: float = 0.85
    crossfade_duration_ms: float = 50.0
    rhythmic: RhythmicSettings = field(default_factory=RhythmicSettings)
    texture: TextureSettings = field(default_factory=TextureSettings)

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        if self.sample_rate not in (22050, 44100, 48000, 96000):
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"Channels must be 1 or 2, got {self.channels}")
        if self.output_format not in ("wav", "flac", "mp3"):
            raise ValueError(f"Unsupported output format: {self.output_format}")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.crossfade_duration_ms < 0:
            raise ValueError("crossfade_duration_ms must be non-negative")
