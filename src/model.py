"""CNN для log-Mel со случайными сдвигами только при обучении."""

from tensorflow import keras


def build_cnn(
    input_shape: tuple[int, int, int] = (64, 313, 1),
    num_classes: int = 8,
) -> keras.Model:
    """Сохранить частотные области после трёх блоков свёрток."""
    time_bins = input_shape[1] // 8
    if input_shape[0] < 8 or time_bins < 1:
        raise ValueError("Вход слишком мал для трёх блоков MaxPooling2D")

    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.10,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="nearest",
                data_format="channels_last",
                seed=42,
                name="spectrogram_shift",
            ),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.AveragePooling2D(
                pool_size=(1, time_bins), name="time_average"
            ),
            keras.layers.Flatten(name="frequency_features"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="speech_emotion_cnn_augmented",
    )