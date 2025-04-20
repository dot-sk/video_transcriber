# Video Transcriber

Автоматизированное скачивание и транскрипция видео с защищенных iframe с использованием mlx-whisper (оптимизировано для Apple Silicon).

## Установка

1.  **Создание и активация виртуального окружения (рекомендуется):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # На Windows используйте `venv\Scripts\activate`
    ```

2.  **Установка зависимостей:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Оптимизация для Apple Silicon:**
    Скрипт использует `mlx-whisper` для оптимизированной транскрипции на устройствах с Apple Silicon. Убедитесь, что зависимости установлены корректно.

4.  **Установка ffmpeg:**
    Whisper требует установки `ffmpeg` в вашей системе.
    - **macOS:** `brew install ffmpeg`
    - **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    - **Windows:** Загрузите с [официального сайта ffmpeg](https://ffmpeg.org/download.html) и добавьте в PATH вашей системы.

## Использование

```bash
python download_and_transcribe.py <page_url> \
  --email <email> \
  --password <password> \
  --model <model_size> \
  --output <video_filename> \
  --transcript <transcript_filename> \
  --login-url <login_url>
```

**Пример:**

```bash
python download_and_transcribe.py https://my.pubnutr.com/some-video-page \
  --email your@email.com \
  --password your_password \
  --model mlx-community/whisper-large-v3-turbo \
  --output downloads/video.mp4 \
  --transcript downloads/transcript.txt
```

**Аргументы:**

- `page_url`: URL страницы, содержащей iframe видеоплеера.
- `--email`: Электронная почта для аутентификации на сайте (обязательный параметр).
- `--password`: Пароль для аутентификации на сайте (обязательный параметр).
- `--model`: Модель Whisper для использования (по умолчанию 'mlx-community/whisper-large-v3-turbo'). Поддерживаются модели из Hugging Face mlx-community.
- `--output`: Имя файла для скачанного видео. По умолчанию создается в директории downloads с использованием названия видео.
- `--transcript`: Имя файла для выходного транскрипта. По умолчанию создается на основе имени видеофайла с расширением .txt.
- `--login-url`: URL-адрес для авторизации (по умолчанию 'https://my.pubnutr.com/cms/system/login').

## Особенности

- Автоматическое извлечение названия видео со страницы
- Аутентификация на сайте для доступа к защищенному контенту
- Оптимизированная транскрипция с использованием mlx-whisper для Apple Silicon
- Автоматическое создание директории downloads для сохранения файлов
- Улучшенное форматирование текста транскрипции
