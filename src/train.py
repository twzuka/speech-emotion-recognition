"""Обучение CNN; опциональные веса классов рассчитываются только по train."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow import keras

from src.model import build_cnn

ROOT = Path(__file__).resolve().parents[1]


def load_split(name: str, settings: dict, splits: dict):
    """Прочитать выборку и проверить её соответствие конфигурации."""
    path = ROOT / "data" / "processed" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as data:
        x, y, actors = data["X"], data["y"], data["actors"]
        if data["emotions"].tolist() != settings["emotions"]:
            raise ValueError(f"{name}: порядок эмоций не совпадает")
        if json.loads(data["settings_json"].item()) != settings:
            raise ValueError(f"{name}: настройки предобработки не совпадают")

    samples = round(settings["sample_rate"] * settings["duration"])
    shape = (settings["n_mels"], samples // settings["hop_length"] + 1, 1)
    if x.ndim != 4 or x.shape[1:] != shape or x.shape[0] == 0:
        raise ValueError(f"{name}: неверная форма X: {x.shape}")
    if y.shape != (len(x),) or actors.shape != y.shape:
        raise ValueError(f"{name}: размеры X, y и actors не согласованы")
    if x.dtype != np.float32 or y.dtype != np.int64:
        raise ValueError(f"{name}: ожидаются X float32 и y int64")
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError(f"{name}: признаки должны быть конечными и в [0, 1]")
    labels = np.arange(len(settings["emotions"]))
    if not np.array_equal(np.unique(y), labels):
        raise ValueError(f"{name}: ожидаются все классы из settings.json")
    if set(actors.tolist()) != set(splits[name]):
        raise ValueError(f"{name}: состав дикторов не совпадает со splits.json")
    return x, y, actors


class ValidationMacroF1(keras.callbacks.Callback):
    """Добавить macro F1 всей validation в журнал каждой эпохи."""

    def __init__(self, x, y, batch_size: int, num_classes: int):
        super().__init__()
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.labels = np.arange(num_classes)

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        for start in range(0, len(self.x), self.batch_size):
            output = self.model(
                self.x[start : start + self.batch_size], training=False
            ).numpy()
            if not np.isfinite(output).all():
                raise ValueError("Неконечные вероятности на validation")
            predictions.append(output.argmax(axis=1))
        score = float(
            f1_score(
                self.y,
                np.concatenate(predictions),
                labels=self.labels,
                average="macro",
                zero_division=0,
            )
        )
        logs["val_macro_f1"] = score
        print(f" — val_macro_f1: {score:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="Заменить максимум эпох")
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Взвешивать ошибки классов обратно пропорционально их числу в train",
    )
    args = parser.parse_args()
    settings = json.loads((ROOT / "config/settings.json").read_text())
    splits = json.loads((ROOT / "config/splits.json").read_text())
    training = json.loads((ROOT / "config/training.json").read_text())
    if args.epochs is not None:
        training["epochs"] = args.epochs
    for key in (
        "epochs",
        "batch_size",
        "early_stopping_patience",
        "reduce_lr_patience",
    ):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"{key} должен быть положительным целым числом")
    if not 0 < training["learning_rate"] < 1:
        raise ValueError("learning_rate должен быть между 0 и 1")
    if not 0 < training["reduce_lr_factor"] < 1:
        raise ValueError("reduce_lr_factor должен быть между 0 и 1")
    if not 0 < training["min_learning_rate"] <= training["learning_rate"]:
        raise ValueError(
            "min_learning_rate должен быть положительным и не выше learning_rate"
        )

    print("Настройки обучения:", json.dumps(training, ensure_ascii=False), flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU не найден. Выполните source activate_gpu.sh")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.utils.set_random_seed(settings["seed"])

    x_train, y_train, train_actors = load_split("train", settings, splits)
    x_val, y_val, val_actors = load_split("validation", settings, splits)
    if set(train_actors.tolist()) & set(val_actors.tolist()):
        raise ValueError("Дикторы train и validation пересекаются")

    num_classes = len(settings["emotions"])
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weight = None
    if args.balance_classes:
        class_weight = {
            label: float(len(y_train) / (num_classes * count))
            for label, count in enumerate(class_counts)
        }
    print(
        "Число записей train по классам:",
        dict(zip(settings["emotions"], class_counts.tolist())),
        flush=True,
    )
    print("Веса классов:", class_weight, flush=True)

    started = datetime.now(timezone.utc)
    run_dir = ROOT / "models" / started.strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "best.keras"
    info = {
        "started_utc": started.isoformat(),
        "settings": settings,
        "training": training,
        "balance_classes": args.balance_classes,
        "train_class_counts": class_counts.tolist(),
        "class_weight": class_weight,
        "epochs_override": args.epochs,
        "input_shape": list(x_train.shape[1:]),
        "selection_metric": "val_macro_f1",
        "train_actors": sorted(set(train_actors.tolist())),
        "validation_actors": sorted(set(val_actors.tolist())),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "status": "started",
    }
    info_path = run_dir / "run_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")

    with tf.device("/GPU:0"):
        model = build_cnn(x_train.shape[1:], len(settings["emotions"]))
        model.compile(
            optimizer=keras.optimizers.Adam(training["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=False,
        )
    callbacks = [
        ValidationMacroF1(
            x_val, y_val, training["batch_size"], len(settings["emotions"])
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=training["reduce_lr_factor"],
            patience=training["reduce_lr_patience"],
            min_lr=training["min_learning_rate"],
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=training["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
    print("Архитектура:", model.name, "Параметров:", model.count_params(), flush=True)
    print("GPU:", gpus)
    print("Train:", x_train.shape, "Validation:", x_val.shape)
    print("Результаты запуска:", run_dir, flush=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        class_weight=class_weight,
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    scores = history.history["val_macro_f1"]
    best_index = int(np.argmax(scores))
    saved_model = keras.models.load_model(model_path, compile=False)
    probe = saved_model(x_val[:2], training=False).numpy()
    if probe.shape != (2, len(settings["emotions"])) or not np.isfinite(probe).all():
        raise ValueError("Сохранённая модель не прошла проверку загрузки")
    info.update(
        {
            "status": "completed",
            "epochs_completed": len(scores),
            "best_epoch": best_index + 1,
            "best_validation_macro_f1": float(scores[best_index]),
        }
    )
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    print(f"Лучшая эпоха: {best_index + 1}")
    print(f"Лучший validation macro F1: {scores[best_index]:.4f}")
    print("Модель:", model_path)
    print("История:", run_dir / "history.csv")
    print("Сохранение и загрузка модели — OK")


if __name__ == "__main__":
    main()
