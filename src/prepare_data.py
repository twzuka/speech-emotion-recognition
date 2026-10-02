import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import audio_to_logmel

ROOT = Path(__file__).resolve().parent.parent


def main():
    settings = json.loads((ROOT / "config/settings.json").read_text(encoding="utf-8"))
    splits = json.loads((ROOT / "config/splits.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(ROOT / "data/processed/metadata.csv")

    if not metadata["path"].is_unique:
        raise ValueError("В metadata.csv повторяются пути файлов")

    if set(metadata["split"]) != set(splits):
        raise ValueError("Выборки не соответствуют splits.json")

    # Проверяем состав дикторов до обработки.
    for split, actors in splits.items():
        rows = metadata[metadata["split"] == split]
        if set(rows["actor"]) != set(actors):
            raise ValueError(f"Неверный состав дикторов: {split}")

    emotion_ids = {name: index for index, name in enumerate(settings["emotions"])}

    samples = round(settings["sample_rate"] * settings["duration"])
    frames = 1 + samples // settings["hop_length"]
    expected_shape = (settings["n_mels"], frames)

    output_dir = ROOT / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        rows = metadata[metadata["split"] == split]
        features = []
        labels = []

        print(f"\nОбработка {split}: {len(rows)} записей", flush=True)

        for index, row in enumerate(rows.itertuples(index=False), start=1):
            mel = audio_to_logmel(ROOT / row.path, settings)

            if mel.shape != expected_shape or not np.isfinite(mel).all():
                raise ValueError(f"Некорректные признаки: {row.path}")

            features.append(mel)
            labels.append(emotion_ids[row.emotion])

            if index % 100 == 0 or index == len(rows):
                print(f"  {index}/{len(rows)}", flush=True)

        # Последнее измерение — один канал изображения для CNN.
        X = np.stack(features)[..., np.newaxis]
        y = np.asarray(labels, dtype=np.int64)

        output = output_dir / f"{split}.npz"
        np.savez_compressed(
            output,
            X=X,
            y=y,
            emotions=np.asarray(settings["emotions"]),
            actors=rows["actor"].to_numpy(dtype=np.int64),
            paths=rows["path"].to_numpy(dtype=str),
            settings_json=json.dumps(settings, ensure_ascii=False),
        )

        print(f"Сохранено {output.name}: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
