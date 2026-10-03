# Speech Emotion Recognition

Учебный проект по классификации выраженных в речи эмоций на датасете RAVDESS. Для распознавания планируется одна небольшая 2D CNN на log-Mel спектрограммах.

## Статус

Реализованы анализ данных, предобработка аудио и подготовка выборок. TensorFlow настроен; выполнение свёртки на NVIDIA RTX 4070 проверено. Обучение CNN, оценка качества и приложение Streamlit — следующие этапы.

## Окружение

Проверенная среда: WSL2, Ubuntu 26.04, Python 3.11.16, TensorFlow 2.21.0 и Keras 3.15.1. Окружение и зависимости управляются через uv.

Все команды выполняются из корня проекта в Bash.

## Установка

Предварительно установите uv. Для запуска с GPU требуется драйвер NVIDIA для Windows с поддержкой WSL2.

```bash
git clone https://github.com/twzuka/speech-emotion-recognition.git
cd speech-emotion-recognition
uv python install 3.11.16
uv venv --python 3.11.16
uv pip sync --python .venv/bin/python requirements.txt
uv pip check --python .venv/bin/python
```

Для запуска с GPU в каждом новом терминале:

```bash
source activate_gpu.sh
.venv/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

`activate_gpu.sh` активирует `.venv` и добавляет каталоги библиотек NVIDIA в `LD_LIBRARY_PATH` текущего терминала. Проверка должна вывести устройство `GPU:0`.

Прямые зависимости заданы в `requirements.in`; точные версии, включая транзитивные зависимости, закреплены в `requirements.txt`. После изменения `requirements.in`:

```bash
uv pip compile --python .venv/bin/python requirements.in -o requirements.txt
uv pip sync --python .venv/bin/python requirements.txt
uv pip check --python .venv/bin/python
```

## Датасет

Используется речевая часть [RAVDESS](https://zenodo.org/records/1188976): `Audio_Speech_Actors_01-24.zip`, 1440 WAV-файлов, 24 диктора, 8 эмоций. Исходные записи: английская речь, 48 кГц, 16 бит.

### Загрузка

Скачайте архив со страницы датасета и сохраните в `data/raw/Audio_Speech_Actors_01-24.zip`.

```bash
mkdir -p data/raw
```

Проверьте контрольную сумму:

```bash
printf '%s\n' 'bc696df654c87fed845eb13823edef8a  data/raw/Audio_Speech_Actors_01-24.zip' | md5sum -c
```

После результата `OK` распакуйте архив:

```bash
.venv/bin/python -m zipfile -e data/raw/Audio_Speech_Actors_01-24.zip data/raw/ravdess
```

### Классы и разделение

Порядок классов закреплён в `config/settings.json`.

| Метка | Эмоция |
|---|---|
| 0 | neutral |
| 1 | calm |
| 2 | happy |
| 3 | sad |
| 4 | angry |
| 5 | fearful |
| 6 | disgust |
| 7 | surprised |

Разделение задано в `config/splits.json`. Дикторы между выборками не пересекаются.

| Выборка | Дикторы | Записи |
|---|---|---|
| train | 01–16 | 960 |
| validation | 17–20 | 240 |
| test | 21–24 | 240 |

Train предназначена для обучения, validation — для подбора параметров, test — для финальной оценки.

## Подготовка данных

После распаковки датасета выполните последовательно:

```bash
.venv/bin/python src/create_metadata.py
.venv/bin/python src/analyze_data.py
.venv/bin/python -m src.prepare_data
```

Результаты: `data/processed/metadata.csv`, `reports/figures/train_data_overview.png` и три архива выборок в `data/processed/`.

### Предобработка

Конвейер реализован в `src/preprocessing.py`; параметры заданы в `config/settings.json`.

| Параметр | Значение |
|---|---|
| Частота дискретизации | 16 000 Гц |
| Каналы | моно |
| Длительность | 5 секунд |
| Mel-полосы | 64 |
| FFT / шаг окна | 1024 / 256 |
| Диапазон log-Mel | 80 дБ относительно максимума записи |
| Масштаб признаков | `[0, 1]` |
| Seed | 42 |

Короткие записи дополняются нулями справа; длинные обрезаются до первых пяти секунд. Автоматическое удаление тишины и аугментация не реализованы.

### Формат NPZ

| Архив | Форма `X` | Форма `y` |
|---|---|---|
| `train.npz` | `(960, 64, 313, 1)` | `(960,)` |
| `validation.npz` | `(240, 64, 313, 1)` | `(240,)` |
| `test.npz` | `(240, 64, 313, 1)` | `(240,)` |

| Поле | Содержимое |
|---|---|
| `X` | Log-Mel спектрограммы, `float32` |
| `y` | Метки классов, `int64` |
| `emotions` | Названия классов в порядке меток |
| `actors` | Номера дикторов для каждой записи |
| `paths` | Пути исходных WAV относительно корня проекта |
| `settings_json` | Снимок настроек предобработки |

## Файлы проекта

| Путь | Назначение |
|---|---|
| `config/` | Параметры обработки и разделение дикторов |
| `src/create_metadata.py` | Создание таблицы метаданных |
| `src/analyze_data.py` | Анализ train и построение графика |
| `src/preprocessing.py` | Преобразование аудио в log-Mel |
| `src/prepare_data.py` | Формирование архивов выборок |
| `tests/test_config.py` | Тесты конфигурации |
| `activate_gpu.sh` | Активация окружения и настройка путей GPU-библиотек |
| `requirements.in` / `requirements.txt` | Прямые зависимости / закреплённые версии |

Каталоги `data/raw/`, `data/processed/`, `.venv/` и `models/` исключены из Git. Датасет и подготовленные массивы необходимо получить отдельно.

## Проверки

```bash
.venv/bin/python -m pytest -q
```

Автоматические тесты проверяют конфигурацию. Дополнительно вручную проверены формы и типы NPZ, конечность и диапазон признаков, метки классов, сохранённые настройки, состав дикторов и отсутствие их пересечений. Свёртка TensorFlow проверена на GPU с отключённым переносом операций на CPU.

## Следующие этапы

Обучение 2D CNN → финальная оценка → модуль предсказания → интерфейс Streamlit для загрузки WAV и отображения вероятностей классов.

Основная метрика — macro F1-score. Дополнительные: accuracy, macro precision/recall, weighted F1 и матрица ошибок. Результатов обучения пока нет.

## Ограничения и источник данных

RAVDESS содержит актёрскую речь на английском языке. Классификация относится к выраженной в записи эмоции; перенос результатов на естественную или русскоязычную речь требует отдельной проверки.

Авторы датасета: Steven R. Livingstone и Frank A. Russo. Лицензия данных: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). [Статья о RAVDESS](https://doi.org/10.1371/journal.pone.0196391).
