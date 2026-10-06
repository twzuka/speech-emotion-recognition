"""Оценка сохранённой CNN на test без дополнительного обучения."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

from src.train import load_split

ROOT = Path(__file__).resolve().parents[1]


def validate_run(info: dict, settings: dict, splits: dict):
    """Проверить настройки запуска и независимость test по дикторам."""
    if info["status"] != "completed":
        raise ValueError("Обучение этого запуска не завершено")
    if info["settings"] != settings:
        raise ValueError("settings.json не совпадает с настройками обучения")
    if info["selection_metric"] != "val_macro_f1":
        raise ValueError("Ожидается модель, выбранная по validation macro F1")
    for name in ("train", "validation"):
        if set(info[f"{name}_actors"]) != set(splits[name]):
            raise ValueError(f"Состав дикторов {name} изменился после обучения")
    groups = [set(splits[name]) for name in ("train", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Дикторы train, validation и test пересекаются")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Папка запуска с best.keras и run_info.json",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()

    settings = json.loads((ROOT / "config/settings.json").read_text())
    splits = json.loads((ROOT / "config/splits.json").read_text())
    info = json.loads((run_dir / "run_info.json").read_text())
    validate_run(info, settings, splits)
    x_test, y_test, actors = load_split("test", settings, splits)
    if list(x_test.shape[1:]) != info["input_shape"]:
        raise ValueError("Форма test не совпадает с входом модели при обучении")

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    model_path = run_dir / "best.keras"
    model = keras.models.load_model(model_path, compile=False)
    emotions = settings["emotions"]
    labels = np.arange(len(emotions))
    if tuple(model.input_shape[1:]) != tuple(x_test.shape[1:]):
        raise ValueError("Модель имеет другую форму входа")
    if tuple(model.output_shape[1:]) != (len(emotions),):
        raise ValueError("Число выходов модели не совпадает с числом эмоций")

    batch_size = info["training"]["batch_size"]
    predictions = []
    for start in range(0, len(x_test), batch_size):
        batch = x_test[start : start + batch_size]
        probabilities = model(batch, training=False).numpy()
        if probabilities.shape != (len(batch), len(emotions)):
            raise ValueError("Некорректная форма вероятностей")
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        ):
            raise ValueError("Некорректные вероятности модели")
        predictions.append(probabilities.argmax(axis=1))
    y_pred = np.concatenate(predictions)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(
        f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    )
    report_args = dict(labels=labels, target_names=emotions, zero_division=0)
    report_text = classification_report(y_test, y_pred, digits=4, **report_args)
    report_dict = classification_report(y_test, y_pred, output_dict=True, **report_args)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    results = {
        "run": run_dir.name,
        "model_path": str(model_path),
        "split": "test",
        "samples": len(y_test),
        "test_actors": sorted(set(actors.tolist())),
        "best_epoch": info["best_epoch"],
        "selection_metric": info["selection_metric"],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "emotions": emotions,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
    }
    report_dir = ROOT / "reports" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "test_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=emotions).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        xticks_rotation=45,
    )
    ax.set_title("CNN: confusion matrix on test")
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print("Модель:", model_path)
    print(f"Test: {len(y_test)} записей; дикторы {results['test_actors']}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(report_text)
    print("Результаты:", report_dir)


if __name__ == "__main__":
    main()
