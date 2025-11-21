from pathlib import Path
from typing import Union

import numpy as np
import librosa
from . import config

AudioPath = Union[str, Path]


def load_audio(path: AudioPath) -> np.ndarray:
    """
    Load an audio file as mono, resampled to config.SAMPLE_RATE.
    Returns a 1D numpy array (float32).
    """
    path = Path(path)
    y, sr = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def _fix_length(y: np.ndarray) -> np.ndarray:
    """
    Crop or pad the waveform to a fixed length defined by SEGMENT_SEC.
    """
    target_len = int(config.SAMPLE_RATE * config.SEGMENT_SEC)

    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width))
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]

    return y


def wav_to_logmel(y: np.ndarray) -> np.ndarray:
    """
    Convert a waveform to a log-Mel spectrogram:
    shape = (NUM_MEL, time_frames), dtype float32.
    """
    y = _fix_length(y)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.NUM_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.NUM_MEL,
        power=2.0,  # power spectrogram (magnitude**2)
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)


def extract_features_from_path(path: AudioPath) -> np.ndarray:
    """
    Convenience function: path → waveform → log-Mel features.
    """
    y = load_audio(path)
    feats = wav_to_logmel(y)
    return feats
