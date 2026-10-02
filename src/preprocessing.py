import librosa
import numpy as np


def audio_to_logmel(path, settings):
    """Преобразует аудио в log-Mel спектрограмму."""

    # Читаем первые пять секунд, переводим в моно и 16000 Гц.
    audio, _ = librosa.load(
        path,
        sr=settings["sample_rate"],
        mono=True,
        duration=settings["duration"],
    )

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Некорректная аудиозапись: {path}")

    if not np.any(audio):
        raise ValueError(f"Запись полностью беззвучна: {path}")

    # Обрезаем лишние отсчёты или добавляем нули справа.
    target_length = round(settings["sample_rate"] * settings["duration"])
    audio = librosa.util.fix_length(audio, size=target_length)

    # Строим mel-спектрограмму мощности.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=settings["sample_rate"],
        n_mels=settings["n_mels"],
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        power=2.0,
        center=True,
        pad_mode="constant",
    )

    # Переводим в децибелы, затем в диапазон от 0 до 1.
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    result = np.clip((log_mel + 80.0) / 80.0, 0.0, 1.0)

    return result.astype(np.float32)
