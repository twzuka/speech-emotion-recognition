"""Формирование data/processed/metadata.csv по WAV-файлам из data/raw/ravdess."""

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/raw/ravdess"
OUTPUT = ROOT / "data/processed/metadata.csv"


def build_metadata(
    data_dir: Path, settings: dict, splits: dict, root: Path = ROOT
) -> pd.DataFrame:
    """Собрать таблицу записей с эмоцией, диктором и выборкой из имён файлов."""
    actor_splits = {
        actor: split for split, actors in splits.items() for actor in actors
    }

    rows = []

    for path in sorted(data_dir.rglob("*.wav")):
        parts = path.stem.split("-")

        # RAVDESS: семь частей имени; 03 — аудио, 01 — речь.
        if len(parts) != 7 or parts[:2] != ["03", "01"]:
            raise ValueError(f"Неожиданное имя файла: {path.name}")

        emotion_id = int(parts[2])
        actor = int(parts[6])

        if not 1 <= emotion_id <= len(settings["emotions"]):
            raise ValueError(f"Неизвестная эмоция: {path.name}")

        if actor not in actor_splits:
            raise ValueError(f"Диктор отсутствует в splits.json: {actor}")

        info = sf.info(path)

        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "actor": actor,
                "emotion": settings["emotions"][emotion_id - 1],
                "split": actor_splits[actor],
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
            }
        )

    if not rows:
        raise ValueError(f"WAV-файлы не найдены в {data_dir}")

    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "config/splits.json").read_text(encoding="utf-8"))

    metadata = build_metadata(DATA_DIR, settings, splits)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUTPUT, index=False)

    print(f"Сохранено: {OUTPUT.relative_to(ROOT)}")
    print(f"Всего записей: {len(metadata)}")

    print("\nКоличество записей по выборкам:")
    print(metadata["split"].value_counts().reindex(splits.keys()))

    print("\nРаспределение эмоций:")
    print(pd.crosstab(metadata["emotion"], metadata["split"]))

    print("\nДлительность записей, секунды:")
    print(metadata["duration"].describe().round(2))

    print("\nЧастоты исходных WAV:", sorted(metadata["sample_rate"].unique()))
    print("Количество каналов:", sorted(metadata["channels"].unique()))


if __name__ == "__main__":
    main()
