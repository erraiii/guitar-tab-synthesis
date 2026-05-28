# Guitar Tab Synthesis

<a href="https://universe.roboflow.com/errai-zca9d/guitar-parts-detection">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg" />
</a>


Проект `guitar-tab-synthesis` конвертирует видео с игрой на гитаре в табулатуру, комбинируя визуальный анализ рук и грифа с аудио-анализом.

## Описание

Система:
- извлекает аудио из видео с помощью `ffmpeg`
- отправляет аудио на сервис распознавания нот, расположенный в каталоге `audio_service`
- детектирует гриф и кисти гитариста на видеокадрах
- сопоставляет аудио- и визуальную информацию
- генерирует табулатуру в формате `txt`, `pdf` или обоих сразу

## Важные требования

1. Python 3.10+.
2. `ffmpeg` должен быть установлен и доступен из PATH.
3. Отдельный аудио-сервис, находящийся в каталоге `audio_service`.
   - Сервис запускается через Docker-контейнер.
   - По умолчанию сервис ожидается на `http://localhost:8000/predict`.
   - Сервис должен принимать POST-запрос с файлом (`file`) и возвращать JSON с полем `notes`.

## Установка

1. Клонируйте репозиторий.
2. Создайте и активируйте виртуальное окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Установите зависимости:

```powershell
pip install -r requirements.txt
```

4. Убедитесь, что `ffmpeg` доступен из PATH:

```powershell
ffmpeg -version
```

## Настройка аудио-сервиса

Сборка и запуск Docker-контейнера:

```powershell
cd audio_service
docker build -t guitar-audio-service .
docker run --rm -p 8000:8000 guitar-audio-service
```

Если вы хотите держать сервис запущенным в фоне, добавьте `-d`:

```powershell
docker run -d --name guitar-audio-service -p 8000:8000 guitar-audio-service
```

Путь к сервису и таймаут заданы в `src/config.py`:

- `AUDIO_SERVICE_URL` — URL для POST-запросов
- `AUDIO_SERVICE_TIMEOUT` — таймаут в секундах
- `AUDIO_CONFIDENCE_THRESHOLD` — порог доверия

Если сервис работает не локально, замените `AUDIO_SERVICE_URL` на нужный адрес.

## Запуск

Основная точка входа — `tab_system/src/main.py`.

Запускайте команды из корня репозитория `guitar-tab-synthesis`.

```powershell
python tab_system/src/main.py path\to\video.mp4
```

Примеры:

```powershell
python tab_system/src/main.py videos\example.mp4
python tab_system/src/main.py videos\example.mp4 --output results --format txt
python tab_system/src/main.py videos\example.mp4 --verbose --log-file
```

## Флаги запуска

`src/main.py` поддерживает следующие опции:

- `-o`, `--output`
  - Каталог для сохранения результатов.
  - По умолчанию: `output/`.

- `--format`
  - Формат вывода.
  - Допустимые значения: `txt`, `pdf`, `both`.
  - По умолчанию: `both`.

- `-v`, `--verbose`
  - Включает подробное логирование (`DEBUG`).

- `-q`, `--quiet`
  - Включает тихий режим (по умолчанию используется уровень `WARNING`).

- `--log-file [PATH]`
  - Сохраняет логи в файл.
  - Если указан без пути, используется `PROJECT_ROOT/tab_synthesis.log`.
  - Если указан путь, логи пишутся в этот файл.

- `--verbose` и `--quiet` нельзя использовать вместе.


## Особенности

- Аудио-сервис должен быть запущен до обработки видео.
- Модель для детекции руки `hand_landmarker.task` загружается автоматически при необходимости.
- Модель для детекции частей гитары `guitar_model.pt` должна находиться в папке `models/`.

## Стек

- Python
- OpenCV
- YOLO
- MediaPipe Hands
- Basic Pitch
- ffmpeg
- Docker


