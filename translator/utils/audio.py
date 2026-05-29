"""Audio utility functions."""

import numpy as np
import soundfile as sf
from pathlib import Path


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 44100):
    """Save audio array to wav file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)
