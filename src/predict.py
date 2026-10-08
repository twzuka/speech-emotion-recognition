"""Распознавание эмоции в отдельном WAV-файле."""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.preprocessing import audio_to_logmel

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    """Разрешить относительный путь от корня проекта."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


class EmotionPredictor:
    """Загрузить модель и использовать её для отдельных записей."""

    def __init__(self, run_dir):
        run_dir = project_path(run_dir)
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        settings = json.loads(
            (ROOT / "config/settings.json").read_text(encoding="utf-8")
        )

        if info["status"] != "completed":
            raise ValueError("Обучение этого запуска не завершено")
        if info["settings"] != settings:
            raise ValueError("Настройки отличаются от настроек обучения")

        self.settings = settings
        self.emotions = settings["emotions"]
        self.model = keras.models.load_model(run_dir / "best.keras", compile=False)

        if tuple(self.model.input_shape[1:]) != tuple(info["input_shape"]):
            raise ValueError("Форма входа модели не совпадает с run_info.json")
        if tuple(self.model.output_shape[1:]) != (len(self.emotions),):
            raise ValueError("Число выходов модели не совпадает с числом эмоций")

    def predict(self, audio_path):
        """Вернуть название эмоции и вероятности всех классов."""
        audio_path = project_path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")
        if audio_path.suffix.lower() != ".wav":
            raise ValueError("Ожидается файл в формате WAV")

        spectrogram = audio_to_logmel(audio_path, self.settings)

        # Добавляем измерения канала и одной записи.
        features = spectrogram[np.newaxis, :, :, np.newaxis]
        if features.shape[1:] != tuple(self.model.input_shape[1:]):
            raise ValueError("Форма спектрограммы не подходит модели")

        output = self.model(features, training=False).numpy()
        if (
            output.shape != (1, len(self.emotions))
            or not np.isfinite(output).all()
            or np.any(output < 0)
            or np.any(output > 1)
            or not np.allclose(output.sum(axis=1), 1, atol=1e-5)
        ):
            raise ValueError("Модель вернула некорректные вероятности")

        probabilities = output[0]
        emotion = self.emotions[int(probabilities.argmax())]
        return emotion, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Папка запуска с best.keras и run_info.json",
    )
    parser.add_argument("--audio", required=True, help="Путь к WAV-файлу")
    args = parser.parse_args()

    predictor = EmotionPredictor(args.run_dir)
    emotion, probabilities = predictor.predict(args.audio)

    print("Файл:", project_path(args.audio))
    print("Предсказанная эмоция:", emotion)
    print("Вероятности модели:")
    for index in np.argsort(-probabilities):
        print(f"  {predictor.emotions[index]:12s}" f" {probabilities[index]:7.2%}")


if __name__ == "__main__":
    main()
