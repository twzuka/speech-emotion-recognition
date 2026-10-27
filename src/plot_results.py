"""Графики запуска по сохранённым отчётам: без обучения, модели и датасета.

Кривые обучения строятся по history.csv, матрица ошибок — по test_metrics.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]

TRAIN_COLOR = "#2a78d6"
VALIDATION_COLOR = "#eb6834"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e1e0d9"
MATRIX_COLORS = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#104281"]
)


def style_axis(ax: plt.Axes, title: str) -> None:
    """Оформить оси: заголовок слева, светлая сетка, без лишних рамок."""
    ax.set_title(title, loc="left", color=TEXT_COLOR)
    ax.set_xlabel("Эпоха", color=MUTED_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR)


def plot_history(history: pd.DataFrame, best_epoch: int, output: Path) -> None:
    """Построить loss, accuracy и validation macro F1 по эпохам."""
    # CSVLogger считает эпохи с нуля, а best_epoch в run_info.json — с единицы.
    epochs = history["epoch"] + 1
    best = history[epochs == best_epoch].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [
        ("loss", "val_loss", "Функция потерь"),
        ("accuracy", "val_accuracy", "Accuracy"),
        (None, "val_macro_f1", "Validation macro F1"),
    ]
    for ax, (train_column, validation_column, title) in zip(axes, panels):
        if train_column:
            ax.plot(
                epochs, history[train_column], color=TRAIN_COLOR, lw=2, label="train"
            )
        ax.plot(
            epochs,
            history[validation_column],
            color=VALIDATION_COLOR,
            lw=2,
            label="validation",
        )
        ax.axvline(best_epoch, color=MUTED_COLOR, lw=1, ls="--")
        style_axis(ax, title)

    axes[0].legend(frameon=False)
    # Место над лучшей точкой для подписи.
    axes[2].set_ylim(0, best["val_macro_f1"] + 0.1)
    axes[2].plot(
        best_epoch,
        best["val_macro_f1"],
        "o",
        ms=8,
        color=VALIDATION_COLOR,
        markeredgecolor="#fcfcfb",
        markeredgewidth=2,
    )
    axes[2].annotate(
        f"лучшая эпоха {best_epoch}: {best['val_macro_f1']:.4f}",
        (best_epoch, best["val_macro_f1"]),
        xytext=(-12, 12),
        textcoords="offset points",
        ha="right",
        color=TEXT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, output: Path) -> None:
    """Построить матрицу ошибок: цвет — доля строки, число — записи."""
    matrix = np.array(metrics["confusion_matrix"])
    emotions = metrics["emotions"]
    # Нормировка по строкам: в neutral 16 записей, в остальных классах по 32.
    shares = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(shares, cmap=MATRIX_COLORS, vmin=0, vmax=1)

    for row in range(len(emotions)):
        for column in range(len(emotions)):
            ax.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if shares[row, column] > 0.5 else TEXT_COLOR,
            )

    ax.set_xticks(range(len(emotions)), emotions, rotation=45, ha="right")
    ax.set_yticks(range(len(emotions)), emotions)
    ax.set_xlabel("Предсказанная эмоция", color=MUTED_COLOR)
    ax.set_ylabel("Истинная эмоция", color=MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(
        f"Test, дикторы 21–24: accuracy {metrics['accuracy']:.2%}, "
        f"macro F1 {metrics['macro_f1']:.4f}\n"
        "Цвет — доля записей строки, число — количество записей",
        loc="left",
        color=TEXT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/20261006-190944-469997"),
        help="Папка с history.csv, run_info.json и test_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs"),
        help="Папка для PNG-файлов",
    )
    args = parser.parse_args()
    report_dir = ROOT / args.report_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(report_dir / "history.csv")
    info = json.loads((report_dir / "run_info.json").read_text(encoding="utf-8"))
    metrics = json.loads((report_dir / "test_metrics.json").read_text(encoding="utf-8"))

    # Отчёты должны относиться к одному и тому же запуску.
    if len(history) != info["epochs_completed"]:
        raise ValueError("Число эпох в history.csv не совпадает с run_info.json")
    if metrics["best_epoch"] != info["best_epoch"]:
        raise ValueError("Лучшая эпоха в test_metrics.json не совпадает")
    matrix = np.array(metrics["confusion_matrix"])
    if not np.isclose(np.trace(matrix) / matrix.sum(), metrics["accuracy"]):
        raise ValueError("Матрица ошибок не согласована с accuracy")

    plot_history(history, info["best_epoch"], output_dir / "training_curves.png")
    plot_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    print("Графики сохранены:", output_dir)


if __name__ == "__main__":
    main()
