"""Интерфейс распознавания эмоций в WAV-файле."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from src.predict import EmotionPredictor

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "models" / "20261006-190944-469997"

EMOTION_NAMES = {
    "neutral": "Нейтральная",
    "calm": "Спокойствие",
    "happy": "Радость",
    "sad": "Грусть",
    "angry": "Злость",
    "fearful": "Страх",
    "disgust": "Отвращение",
    "surprised": "Удивление",
}


@st.cache_resource
def load_predictor():
    """Сохранить загруженную модель в памяти приложения."""
    return EmotionPredictor(RUN_DIR)


def main():
    st.set_page_config(
        page_title="Распознавание эмоций",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("Распознавание эмоций по голосу")
    st.caption("Учебная модель, обученная на актёрской речи " "на английском языке.")
    st.write(
        "Загрузите WAV-файл с речью. "
        "Для распознавания используются первые пять секунд записи."
    )

    uploaded = st.file_uploader("Выберите аудиозапись", type=["wav"])
    if uploaded is None:
        return

    audio_bytes = uploaded.getvalue()
    if not audio_bytes:
        st.error("Файл пустой. Выберите другую запись.")
        return

    st.audio(audio_bytes, format="audio/wav")

    if not st.button("Распознать эмоцию", type="primary"):
        return

    try:
        with st.spinner("Обработка записи…"):
            predictor = load_predictor()

            # Временный WAV удаляется после обработки.
            with TemporaryDirectory() as directory:
                audio_path = Path(directory) / "recording.wav"
                audio_path.write_bytes(audio_bytes)
                emotion, probabilities = predictor.predict(audio_path)

    except Exception as error:
        message = str(error).strip()

        if "Запись полностью беззвучна" in message:
            message = "В записи только тишина. Выберите WAV-файл с речью."
        elif not message:
            message = (
                "Не удалось прочитать WAV-файл. "
                "Файл повреждён или имеет неподдерживаемый формат."
            )

        st.error(message)
        return

    st.subheader("Результат")
    st.metric("Предсказанная эмоция", EMOTION_NAMES[emotion])
    st.write(f"Вероятность выбранного класса: {probabilities.max():.2%}")

    results = pd.DataFrame(
        {
            "Эмоция": [EMOTION_NAMES[name] for name in predictor.emotions],
            "Вероятность, %": probabilities * 100,
        }
    )
    results = results.sort_values("Вероятность, %", ascending=False).reset_index(
        drop=True
    )

    st.subheader("Вероятности классов")
    st.bar_chart(
        results,
        x="Эмоция",
        y="Вероятность, %",
        horizontal=True,
        sort=False,
    )
    st.dataframe(results.round(2), hide_index=True)

    st.caption(
        "Вероятности отражают выход модели для этой записи. "
        "Правильность предсказания не гарантируется."
    )


if __name__ == "__main__":
    main()
