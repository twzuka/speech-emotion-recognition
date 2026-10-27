"""Тесты разбора имён файлов RAVDESS и проверки разбиения по дикторам."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.create_metadata import build_metadata
from src.prepare_data import check_metadata

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
SPLITS = json.loads((ROOT / "config/splits.json").read_text(encoding="utf-8"))


def make_wav(data_dir, name):
    """Создать короткий WAV 48 кГц в папке актёра, как в архиве RAVDESS."""
    actor = int(Path(name).stem.split("-")[-1])
    path = data_dir / f"Actor_{actor:02d}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(4800), 48000)
    return path


def test_metadata_from_ravdess_names(tmp_path):
    data_dir = tmp_path / "ravdess"
    # Модальность-канал-эмоция-интенсивность-фраза-повтор-актёр.
    make_wav(data_dir, "03-01-01-01-01-01-01.wav")
    make_wav(data_dir, "03-01-05-02-01-01-17.wav")
    make_wav(data_dir, "03-01-08-01-02-02-24.wav")

    metadata = build_metadata(data_dir, SETTINGS, SPLITS, root=tmp_path)

    assert metadata[["actor", "emotion", "split"]].values.tolist() == [
        [1, "neutral", "train"],
        [17, "angry", "validation"],
        [24, "surprised", "test"],
    ]
    assert metadata["path"][0] == "ravdess/Actor_01/03-01-01-01-01-01-01.wav"
    assert metadata["sample_rate"].tolist() == [48000] * 3
    assert metadata["duration"].tolist() == pytest.approx([0.1] * 3)


@pytest.mark.parametrize(
    "name",
    [
        "02-01-01-01-01-01-01.wav",  # видео
        "03-02-01-01-01-01-01.wav",  # пение
        "03-01-09-01-01-01-01.wav",  # эмоции 09 нет
        "03-01-01-01-01-01.wav",  # шесть частей имени
    ],
)
def test_unexpected_names_are_rejected(tmp_path, name):
    data_dir = tmp_path / "ravdess"
    path = data_dir / name
    path.parent.mkdir(parents=True)
    sf.write(path, np.zeros(4800), 48000)

    with pytest.raises(ValueError):
        build_metadata(data_dir, SETTINGS, SPLITS, root=tmp_path)


def test_actor_missing_from_splits_is_rejected(tmp_path):
    data_dir = tmp_path / "ravdess"
    make_wav(data_dir, "03-01-01-01-01-01-24.wav")
    splits = {**SPLITS, "test": [21, 22, 23]}

    with pytest.raises(ValueError, match="Диктор отсутствует"):
        build_metadata(data_dir, SETTINGS, splits, root=tmp_path)


def metadata_for(actor_splits):
    """Таблица с одной записью на каждого диктора."""
    return pd.DataFrame(
        {
            "path": [f"Actor_{actor:02d}.wav" for actor in actor_splits],
            "actor": list(actor_splits),
            "split": list(actor_splits.values()),
        }
    )


def config_actor_splits():
    return {actor: split for split, actors in SPLITS.items() for actor in actors}


def test_check_metadata_accepts_config_split():
    check_metadata(metadata_for(config_actor_splits()), SPLITS)


def test_check_metadata_rejects_actor_leak():
    actor_splits = config_actor_splits()
    # Диктор validation случайно попал в train.
    actor_splits[17] = "train"

    with pytest.raises(ValueError, match="Неверный состав дикторов"):
        check_metadata(metadata_for(actor_splits), SPLITS)


def test_check_metadata_rejects_duplicate_paths():
    metadata = metadata_for(config_actor_splits())
    metadata = pd.concat([metadata, metadata.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="повторяются пути"):
        check_metadata(metadata, SPLITS)
