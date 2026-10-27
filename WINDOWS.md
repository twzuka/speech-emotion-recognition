# Запуск в Windows 11 на CPU

Этот пакет запускает сохранённую модель `20261006-190944-469997`.
WSL, CUDA, исходный датасет и повторное обучение для запуска не нужны.

1. Скачайте из GitHub Release файл `speech-emotion-recognition-windows-cpu.zip`.
   Распакуйте его целиком, например в `C:\Projects\speech-emotion-recognition`.
   Не запускайте файлы прямо внутри ZIP.
2. Один раз установите **Python 3.11 x64 для Windows** с pip и Python Launcher.
   Можно выполнить в PowerShell:

   ```powershell
   winget install --exact --id Python.Python.3.11 --source winget
   ```

   Альтернатива: [официальный установщик Python 3.11.9 x64](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe).
   Проверка в новом PowerShell: `py -3.11 --version`.
   Python 3.14 для этого набора зависимостей не подходит.
3. В PowerShell перейдите в распакованную папку и создайте окружение:

   ```powershell
   cd C:\Projects\speech-emotion-recognition
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-windows.txt
   .\.venv\Scripts\python.exe -m pip check
   .\.venv\Scripts\python.exe check_model.py
   ```

   На первом запуске нужен интернет и свободное место для Python и зависимостей.
   `check_model.py` проверяет контрольные суммы модели и вычисление на CPU.
4. Запустите приложение:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
   ```

5. Откройте <http://127.0.0.1:8501>, загрузите WAV с речью и нажмите
   «Распознать эмоцию». Модель использует первые пять секунд записи.
6. Для остановки нажмите Ctrl+C. Для следующего запуска откройте PowerShell
   в папке проекта и повторите только команду `streamlit run` из пункта 4.

Зависимости устанавливаются только внутри папки проекта; повторные запуски
с тем же списком зависимостей работают без скачивания пакетов.
Окружение `.venv` между компьютерами переносить не нужно.
Сохраняйте запуск в отдельной Windows-папке, не поверх окружения WSL.

Если `.venv` создана другой версией Python, переименуйте её в `.venv-old`
и повторите команды создания окружения из пункта 3.
Не удаляйте окружение домашнего проекта в WSL.
Если при импорте TensorFlow возникает ошибка DLL, установите
[Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe)
и повторите запуск.

В пакете находятся `models/20261006-190944-469997/best.keras` и `run_info.json`.
Это неизменённые файлы домашнего ПК. Для отдельной копии репозитория их также
можно скачать из Release и поместить по этому пути. Автоматические архивы GitHub
`Source code (zip)` / `Source code (tar.gz)` модель и Windows-запуск не содержат;
выбирайте именованный ZIP из первого пункта.

Проверенные версии: Python 3.11.16, TensorFlow 2.21.0, Keras 3.15.1,
Streamlit 1.64.0. Остальные версии закреплены в `requirements-windows.txt`.
Прямые зависимости описаны в `requirements-windows.in`; при сборке использованы
ограничения версий из исходного `requirements.txt`, без NVIDIA/CUDA и Jupyter.

Качество модели не изменено: test accuracy 103/240 = 42.92%, macro F1 0.4057.
Это учебная модель для актёрской речи на английском языке.
