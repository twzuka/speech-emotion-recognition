"""Тесты общей предобработки audio_to_logmel на синтетических WAV."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing import audio_to_logmel

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))

# 5 с × 16 000 Гц / шаг 256 + 1 = 313 кадров.
EXPECTED_SHAPE = (64, 313)


def write_noise(path, seconds, sample_rate=48000, channels=1):
    """Записать WAV со случайным шумом и вернуть путь к нему."""
    rng = np.random.default_rng(42)
    shape = (int(seconds * sample_rate), channels)
    sf.write(path, 0.1 * rng.standard_normal(shape), sample_rate)
    return path


def test_output_shape_type_and_range(tmp_path):
    mel = audio_to_logmel(write_noise(tmp_path / "noise.wav", 3), SETTINGS)

    assert mel.shape == EXPECTED_SHAPE
    assert mel.dtype == np.float32
    assert np.isfinite(mel).all()
    assert mel.min() >= 0 and mel.max() <= 1


@pytest.mark.parametrize("seconds", [1, 5, 8])
def test_any_duration_gives_same_shape(tmp_path, seconds):
    mel = audio_to_logmel(write_noise(tmp_path / "noise.wav", seconds), SETTINGS)

    assert mel.shape == EXPECTED_SHAPE


def test_stereo_is_converted_to_mono(tmp_path):
    path = write_noise(tmp_path / "stereo.wav", 2, channels=2)

    assert audio_to_logmel(path, SETTINGS).shape == EXPECTED_SHAPE


def test_short_record_is_padded_with_silence(tmp_path):
    mel = audio_to_logmel(write_noise(tmp_path / "short.wav", 1), SETTINGS)

    # Первая секунда занимает около 63 кадров, дальше только нули.
    assert mel[:, :60].max() > 0
    assert np.all(mel[:, 100:] == 0)


def test_only_first_five_seconds_are_used(tmp_path):
    long_path = write_noise(tmp_path / "long.wav", 8)
    audio, sample_rate = sf.read(long_path)
    sf.write(tmp_path / "first.wav", audio[: 5 * sample_rate], sample_rate)

    long_mel = audio_to_logmel(long_path, SETTINGS)
    first_mel = audio_to_logmel(tmp_path / "first.wav", SETTINGS)

    np.testing.assert_allclose(long_mel, first_mel, atol=1e-6)


def test_same_file_gives_same_features(tmp_path):
    path = write_noise(tmp_path / "noise.wav", 2)

    np.testing.assert_array_equal(
        audio_to_logmel(path, SETTINGS), audio_to_logmel(path, SETTINGS)
    )


def test_silent_record_is_rejected(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000), 16000)

    with pytest.raises(ValueError, match="беззвучна"):
        audio_to_logmel(path, SETTINGS)
