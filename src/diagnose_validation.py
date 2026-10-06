"""Проверка сохранённой CNN на train и validation без обучения."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from tensorflow import keras

from src.evaluate import validate_run
from src.train import load_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    settings = json.loads((ROOT / "config/settings.json").read_text())
    splits = json.loads((ROOT / "config/splits.json").read_text())
    info = json.loads((run_dir / "run_info.json").read_text())
    validate_run(info, settings, splits)

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = keras.models.load_model(run_dir / "best.keras", compile=False)
    emotions = settings["emotions"]
    labels = list(range(len(emotions)))

    print("Модель:", run_dir / "best.keras")
    print("Архитектура:", model.name)
    print("Параметров:", model.count_params())
    print("Лучшая эпоха:", info["best_epoch"])

    for split in ("train", "validation"):
        x, y, actors = load_split(split, settings, splits)
        probabilities = np.concatenate(
            [
                model(x[start : start + 32], training=False).numpy()
                for start in range(0, len(x), 32)
            ],
            axis=0,
        )
        if probabilities.shape != (len(y), len(emotions)):
            raise ValueError("Неверная форма предсказаний")
        if not np.isfinite(probabilities).all():
            raise ValueError("Предсказания содержат NaN или Inf")
        predicted = probabilities.argmax(axis=1)

        print(f"\n{split.upper()}: {len(y)} записей")
        print(
            classification_report(
                y,
                predicted,
                labels=labels,
                target_names=emotions,
                digits=4,
                zero_division=0,
            )
        )
        if split == "validation":
            score = f1_score(
                y, predicted, labels=labels, average="macro", zero_division=0
            )
            print(f"Macro F1 сейчас: {score:.10f}")
            print(
                "Macro F1 в run_info:",
                f"{info['best_validation_macro_f1']:.10f}",
            )
            print("Accuracy по дикторам validation:")
            for actor in np.unique(actors):
                mask = actors == actor
                correct = int(np.sum(predicted[mask] == y[mask]))
                count = int(mask.sum())
                print(
                    f"Actor {int(actor):02d}: {correct}/{count} "
                    f"({correct / count:.2%})"
                )


if __name__ == "__main__":
    main()
