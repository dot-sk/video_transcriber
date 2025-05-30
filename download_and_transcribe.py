import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
import mlx_whisper
import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --- Helper Functions ---

def extract_video_title(soup: BeautifulSoup) -> str:
    """
    Извлекает название видео из элемента h2 с классом 'lesson-title-value'.

    Args:
        soup: Объект BeautifulSoup с HTML содержимым страницы.

    Returns:
        Название видео или "video" если элемент не найден.
    """
    title_element = soup.select_one('h2.lesson-title-value')
    if title_element and title_element.text.strip():
        # Очищаем название от недопустимых символов в имени файла
        title = title_element.text.strip()
        title = re.sub(r'[\\/*?:"<>|]', "_", title)  # Замена недопустимых символов
        logging.info(f"Найдено название видео: {title}")
        return title
    else:
        logging.warning("Не удалось найти название видео, используем значение по умолчанию")
        return "video"


def authenticate(login_url: str, credentials: Tuple[str, str]) -> requests.Session:
    """
    Аутентифицируется на сайте и возвращает сессию с сохраненными куки.

    Args:
        login_url: URL страницы входа.
        credentials: Кортеж (email, пароль).

    Returns:
        Аутентифицированная сессия requests.Session.

    Raises:
        requests.RequestException: При ошибке сетевых запросов.
    """
    logging.info(f"Аутентификация на {login_url}")
    email, password = credentials

    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # Получаем страницу входа для извлечения параметров аутентификации
    try:
        login_page_response = session.get(login_url, headers=headers, timeout=30)
        login_page_response.raise_for_status()

        # Парсим HTML
        soup = BeautifulSoup(login_page_response.text, "html.parser")

        # Извлекаем параметры из тегов script (особенно в head)
        script_params = extract_script_params(soup)

        # Находим форму входа
        xdget_id = extract_form_id(soup)

        # Создаем данные для аутентификации
        auth_data = create_auth_data(
            soup,
            script_params,
            xdget_id,
            email,
            password,
            login_url
        )

        logging.info("Параметры аутентификации успешно извлечены из страницы")

    except requests.RequestException as e:
        logging.error(f"Не удалось получить страницу входа: {e}")
        raise

    # Отправляем запрос на вход
    try:
        logging.info("Отправка запроса аутентификации...")
        response = session.post(login_url, data=auth_data, headers=headers, timeout=30)
        response.raise_for_status()

        # Проверяем успешность аутентификации
        if check_auth_success(response, session):
            logging.info("Аутентификация успешна")
        else:
            logging.warning("Аутентификация может быть не выполнена, но продолжаем работу")

        return session
    except requests.RequestException as e:
        logging.error(f"Ошибка аутентификации: {e}")
        raise


def extract_script_params(soup: BeautifulSoup) -> Dict[str, str]:
    """Извлекает параметры из script-тегов."""
    script_params = {}

    # Список параметров для поиска
    param_names = [
        "csrfToken", "requestSimpleSign", "gcSessionId",
        "accountId", "controllerId", "actionId",
        "UShort", "ULong", "gcUniqId"
    ]

    # Сначала ищем в head
    head = soup.find("head")
    if head:
        scripts = head.find_all("script")
    else:
        scripts = []

    # Если не найдено в head, ищем во всех script-тегах
    if not scripts:
        scripts = soup.find_all("script")

    # Извлекаем параметры из всех скриптов
    for script in scripts:
        if not script.string:
            continue

        for param in param_names:
            if param in script_params:
                continue  # Уже нашли этот параметр

            pattern = rf'window\.{param}\s*=\s*["\']([^"\']+)["\']'
            match = re.search(pattern, script.string)
            if match:
                script_params[param] = match.group(1)

    if script_params:
        logging.info(f"Найдены параметры в скриптах: {', '.join(script_params.keys())}")

    return script_params


def extract_form_id(soup: BeautifulSoup) -> str:
    """Извлекает ID формы входа."""
    # Ищем форму по классу xdget-loginUserForm
    login_form = soup.find("form", {"class": lambda c: c and "xdget-loginUserForm" in c.split()})

    # Если не найдена, пробуем найти любую форму
    if not login_form:
        login_form = soup.find("form")

    # Если форма найдена, извлекаем data-xdget-id
    if login_form:
        xdget_id = login_form.get("data-xdget-id")

        # Если нет data-xdget-id, пробуем извлечь из id (формат: xdgetXXXXX)
        if not xdget_id:
            form_id = login_form.get("id", "")
            id_match = re.search(r'xdget(\d+)', form_id)
            if id_match:
                xdget_id = id_match.group(1)
            else:
                xdget_id = "99945"  # Значение по умолчанию
    else:
        xdget_id = "99945"  # Значение по умолчанию

    logging.info(f"ID формы входа: {xdget_id}")
    return xdget_id


def create_auth_data(soup: BeautifulSoup, script_params: Dict[str, str],
                     xdget_id: str, email: str, password: str,
                     login_url: str) -> Dict[str, str]:
    """Создаёт словарь с данными для аутентификации."""
    current_time = int(time.time())

    # Основные параметры аутентификации
    auth_data = {
        "action": "processXdget",
        "xdgetId": xdget_id,
        "requestTime": str(current_time),
    }

    # Определяем формат имен параметров: params[x] или params.x
    params_format = "params[%s]"
    if soup.find(string=re.compile(r'params\.email')):
        params_format = "params.%s"

    # Добавляем основные параметры формы
    auth_data.update({
        params_format % "action": "login",
        params_format % "url": login_url + "?required=true",
        params_format % "email": email,
        params_format % "password": password,
        params_format % "null": "",
        params_format % "object_type": "cms_page",
        params_format % "object_id": "-1",
    })

    # Добавляем параметры из скриптов
    for key, value in script_params.items():
        if key == "csrfToken":
            auth_data["_csrf"] = value
        elif key == "requestSimpleSign":
            auth_data["requestSimpleSign"] = value
        elif key in ["gcSessionId", "accountId", "controllerId", "actionId",
                     "UShort", "ULong", "gcUniqId"]:
            auth_data[key] = value

    # Если requestSimpleSign не найден, используем значение по умолчанию
    if "requestSimpleSign" not in auth_data:
        auth_data["requestSimpleSign"] = "d77de075a31ade6131df53dd1fe3c61b"

    # Создаём параметры сессии
    gc_session_id = script_params.get("gcSessionId", str(current_time))

    auth_data.update({
        "gcSession": json.dumps({
            "id": int(gc_session_id),
            "last_activity": time.strftime("%Y-%m-%d+%H:%M:%S"),
            "user_id": 0,
            "utm_id": None
        }),
        "gcVisit": json.dumps({
            "id": current_time + 1,
            "sid": int(gc_session_id)
        }),
        "gcVisitor": json.dumps({
            "id": current_time - 1000,
            "sfix": 1
        }),
        "gcSessionHash": script_params.get("gcSessionHash", "generated_session_hash")
    })

    # Добавляем скрытые поля из формы
    login_form = soup.find("form")
    if login_form:
        for input_tag in login_form.find_all("input", {"type": "hidden"}):
            name = input_tag.get("name")
            value = input_tag.get("value", "")
            if name and name not in auth_data:
                auth_data[name] = value

    return auth_data


def check_auth_success(response: requests.Response, session: requests.Session) -> bool:
    """Проверяет успешность аутентификации."""
    # Проверяем наличие куки
    important_cookies = ["_csrf", "PHPSESSID5", "dd_bdfhyr", "user_session", "auth_token"]
    found_cookies = [cookie for cookie in important_cookies if cookie in session.cookies]

    if found_cookies:
        return True

    # Проверяем содержимое страницы
    soup = BeautifulSoup(response.text, "html.parser")

    # Ищем признаки успешного входа
    user_elements = soup.find_all(string=re.compile(r'(welcome|dashboard|profile|logout|Здравствуйте)', re.I))
    if user_elements:
        return True

    # Проверяем наличие формы для авторизованных пользователей
    logined_form = soup.find("div", {"class": lambda c: c and "logined-form" in c.split()})
    if logined_form:
        return True

    # Проверяем сообщения об ошибках
    error_elements = soup.find_all(string=re.compile(r'(invalid|failed|incorrect|wrong password)', re.I))
    if error_elements:
        logging.error(f"Ошибка аутентификации: {error_elements[0].strip()}")
        return False

    # Если нет явных признаков ошибки, считаем авторизацию успешной
    return True


def extract_master_playlist_urls(page_url: str, session: Optional[requests.Session] = None) -> List[Tuple[str, str]]:
    """
    Extracts all master playlist URLs (m3u8) from iframes embedded in a page.

    Args:
        page_url: The URL of the page containing the video player iframes.
        session: Optional authenticated session for sites requiring login.

    Returns:
        List of tuples containing (master_playlist_url, video_title) for each video found.

    Raises:
        ValueError: If no playlist URLs can be found.
        requests.RequestException: If network requests fail.
    """
    try:
        logging.info(f"Fetching main page: {page_url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Use provided session or create a new one
        if session is None:
            session = requests.Session()

        response = session.get(page_url, headers=headers, timeout=30)
        response.raise_for_status()
        logging.info("Main page fetched successfully.")

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all elements with data-iframe-src attribute using CSS selector
        video_divs = soup.select('[data-iframe-src]')

        if not video_divs:
            raise ValueError(f"Could not find any elements with 'data-iframe-src' attribute on page {page_url}")

        logging.info(f"Found {len(video_divs)} video(s) on the page")

        results = []

        for i, video_div in enumerate(video_divs, 1):
            try:
                # Extract video title for each video (try to find specific title or use generic one)
                video_title = extract_video_title_for_element(soup, i)

                iframe_src_rel = video_div.get("data-iframe-src")
                logging.info(f"Video {i}: Found iframe source: {iframe_src_rel}")

                iframe_src = urljoin(page_url, iframe_src_rel)
                logging.info(f"Video {i}: Complete iframe URL: {iframe_src}")

                # Fetch iframe content using the same session to maintain cookies
                logging.info(f"Video {i}: Fetching iframe content...")
                iframe_response = session.get(iframe_src, headers={"Referer": page_url, **headers}, timeout=30)
                iframe_response.raise_for_status()
                iframe_content = iframe_response.text
                logging.info(f"Video {i}: Iframe content fetched successfully.")

                # Простой поиск masterPlaylistUrl с помощью регулярного выражения
                url_match = re.search(r'"masterPlaylistUrl"\s*:\s*"([^"]+)"', iframe_content)
                if not url_match:
                    # Альтернативный поиск с другим возможным форматом
                    url_match = re.search(r'masterPlaylistUrl\s*=\s*[\'"]([^\'"]+)[\'"]', iframe_content)

                if not url_match:
                    logging.warning(f"Video {i}: Could not find masterPlaylistUrl in iframe content, skipping")
                    continue

                master_playlist_url = url_match.group(1)

                # Unescape URL если в ней есть экранированные символы
                master_playlist_url = master_playlist_url.replace('\\/', '/')

                logging.info(f"Video {i}: Successfully extracted master playlist URL: {master_playlist_url}")
                results.append((master_playlist_url, video_title))

            except Exception as e:
                logging.error(f"Video {i}: Error processing video: {e}")
                continue

        if not results:
            raise ValueError("Could not extract any valid playlist URLs from the page")

        logging.info(f"Successfully extracted {len(results)} video(s)")
        return results

    except requests.RequestException as e:
        logging.error(f"Network error during extraction: {e}")
        raise
    except Exception as e:
        logging.error(f"Error extracting playlist URLs: {e}")
        raise ValueError(f"Extraction failed: {e}") from e


def extract_video_title_for_element(soup: BeautifulSoup, video_index: int) -> str:
    """
    Извлекает название видео для конкретного элемента видео по индексу.

    Args:
        soup: Объект BeautifulSoup с HTML содержимым страницы.
        video_div: Элемент video div.
        video_index: Индекс видео на странице (начиная с 1).

    Returns:
        Название видео или "video_{index}" если заголовок не найден.
    """
    title = None

    title_element = soup.select_one('h2.lesson-title-value')
    if title_element and title_element.text.strip():
        # Если на странице несколько видео, добавляем индекс
        title = title_element.text.strip()
        if video_index > 1:
            title = f"{title}_часть_{video_index}"

    # Если не нашли, используем значение по умолчанию
    if not title:
        title = f"video_part_{video_index}"

    # Очищаем название от недопустимых символов в имени файла
    title = re.sub(r'[\\/*?:"<>|]', "_", title)
    logging.info(f"Video {video_index}: Extracted title: {title}")
    return title


def download_with_ytdlp(url: str, filename: str, session: Optional[requests.Session] = None, format: str = "ba/b[height<=360]") -> None:
    """
    Downloads video from a URL (likely m3u8 playlist) using yt-dlp.

    Args:
        url: The URL to download from.
        filename: The output filename (path).
        session: Optional authenticated session for sites requiring login.
        format: Format selection string for yt-dlp (default: audio-only or low-res video).

    Raises:
        yt_dlp.utils.DownloadError: If download fails.
    """
    logging.info(f"Starting download from {url} to {filename}")
    logging.info(f"Using format selection: {format}")

    # Extract cookies from session if available
    cookies = None
    if session:
        cookies = {name: value for name, value in session.cookies.items()}
        logging.info(f"Using authenticated session with cookies: {', '.join(cookies.keys())}")

    ydl_opts = {
        'outtmpl': filename,
        'format': format, # Use provided format selection
        'quiet': False,
        'progress': True,
        'noplaylist': True, # Ensure only single video is downloaded if URL points to one
    }

    # Add cookies if available
    if cookies:
        ydl_opts['cookiefile'] = None  # Don't use a cookie file
        ydl_opts['cookiesfrombrowser'] = None  # Don't use browser cookies
        ydl_opts['cookies'] = cookies  # Use our session cookies directly

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logging.info(f"Successfully downloaded video to {filename}")
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"yt-dlp download failed: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during download: {e}")
        raise


def transcribe_whisper(input_video: str, model_size: str, out_txt: str) -> None:
    """
    Transcribes the audio from a video file using mlx-whisper for optimal performance
    on Apple Silicon.

    Args:
        input_video: Path to the input video file.
        model_size: The Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        out_txt: Path to save the transcript text file.

    Raises:
        FileNotFoundError: If input video file not found.
        ImportError: If mlx_whisper is not installed or ffmpeg is missing.
        Exception: If transcription fails.
    """
    input_path = Path(input_video)
    output_path = Path(out_txt)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    logging.info(f"Using mlx-whisper for transcription (Optimized for Apple Silicon)")
    logging.info(f"Loading model: {model_size} (will download if needed)")

    try:
        # Загрузка модели (mlx-whisper сам скачает, если нужно)
        # model = mlx_whisper.load_model(model_size) # Загрузка модели не нужна явно в 0.4.2

        logging.info(f"Starting transcription for {input_video}...")

        # Параметры транскрипции для mlx_whisper
        # Убираем beam_size и best_of, так как beam search не реализован
        transcription_options = {
            "language": "ru",           # Язык распознавания
            # "fp16": True,             # MLX обычно использует fp16 по умолчанию
            # "beam_size": 5,             # Убрано: Beam search не реализован
            # "best_of": 5,               # Убрано: Beam search не реализован
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), # Кортеж температур
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": True,
            # "patience": 1.0,            # Убрано: связано с beam search
            "verbose": True,            # Включаем подробный вывод
            "initial_prompt": "Далее идет транскрипция видео на русском языке."
        }

        # Выполняем транскрипцию
        # mlx_whisper.transcribe возвращает словарь, похожий на оригинальный whisper
        result = mlx_whisper.transcribe(str(input_path), path_or_hf_repo=model_size, **transcription_options)

        logging.info(f"Transcription completed successfully using mlx-whisper.")

        # Постобработка транскрипции для улучшения читаемости
        transcript = result["text"]
        transcript = improve_formatting(transcript)
        logging.info("Transcription complete with post-processing.")

        logging.info(f"Saving transcript to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        logging.info("Transcript saved successfully.")

    except ImportError:
         logging.error("mlx-whisper or its dependency (like ffmpeg) not found. Please install with 'pip install mlx-whisper' and ensure ffmpeg is installed ('brew install ffmpeg').")
         raise
    except Exception as e:
        logging.error(f"mlx-whisper transcription failed: {e}")
        raise


def improve_formatting(text: str) -> str:
    """
    Улучшает форматирование транскрипции для лучшей читаемости.

    Args:
        text: Исходный текст транскрипции.

    Returns:
        Отформатированный текст.
    """
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Добавляем точки в конце предложений, если их нет
    text = re.sub(r'([а-яА-Яa-zA-Z0-9])\s+([А-ЯA-Z])', r'\1. \2', text)

    # Удаляем повторяющиеся знаки препинания
    text = re.sub(r'([.,!?])\1+', r'\1', text)

    # Убираем пробелы перед знаками препинания
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # Добавляем перенос строки после длинных предложений
    text = re.sub(r'([.!?]) ', r'\1\n', text)

    return text


# --- Main Execution ---

def main():
    """Main function to orchestrate the download and transcription process."""
    parser = argparse.ArgumentParser(
        description="Download a video from a protected iframe and transcribe it using mlx-whisper (optimized for Apple Silicon)."
    )
    parser.add_argument(
        "page_url", type=str, help="URL of the page containing the video player iframe."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="Whisper model size to use (default: mlx-community/whisper-large-v3-turbo). @see https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Filename for the downloaded video (default: based on video title).",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Filename for the output transcript (default: based on video title).",
    )
    parser.add_argument(
        "--login-url",
        type=str,
        default="https://my.pubnutr.com/cms/system/login",
        help="URL for authentication (if site requires login).",
    )
    # Required login credentials
    parser.add_argument(
        "--email",
        type=str,
        required=True,
        help="Email/login for site authentication.",
    )
    parser.add_argument(
        "--password",
        type=str,
        required=True,
        help="Password for site authentication.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="ba/b[height<=480]",
        help="Format selection for yt-dlp (default: ba/b[height<=480] - audio-only or low-res video).",
    )

    args = parser.parse_args()

    try:
        # Создаем директорию для загрузок, если её нет
        downloads_dir = Path("./downloads")
        downloads_dir.mkdir(exist_ok=True)

        # Authenticate with provided credentials
        session = authenticate(args.login_url, (args.email, args.password))

        # 1. Extract the master playlist URLs and video titles using the authenticated session
        master_playlist_urls = extract_master_playlist_urls(args.page_url, session)

        # Определяем имена файлов на основе заголовка видео
        if not master_playlist_urls:
            raise ValueError("No valid playlist URLs found")

        # Убедимся, что директории для вывода существуют
        for master_playlist_url, video_title in master_playlist_urls:
            output_video = downloads_dir / f"{video_title}.mp4"
            output_transcript = downloads_dir / f"{video_title}.txt"
            output_video.parent.mkdir(parents=True, exist_ok=True)
            output_transcript.parent.mkdir(parents=True, exist_ok=True)

        # 2. Download the videos using yt-dlp with the authenticated session
        for master_playlist_url, video_title in master_playlist_urls:
            output_video = downloads_dir / f"{video_title}.mp4"
            download_with_ytdlp(master_playlist_url, str(output_video), session, args.format)

        # 3. Transcribe the videos using mlx-whisper
        for master_playlist_url, video_title in master_playlist_urls:
            output_video = downloads_dir / f"{video_title}.mp4"
            output_transcript = downloads_dir / f"{video_title}.txt"
            transcribe_whisper(str(output_video), args.model, str(output_transcript))

        logging.info("Process completed successfully!")
        logging.info(f"Processed {len(master_playlist_urls)} video(s):")
        for master_playlist_url, video_title in master_playlist_urls:
            output_video = downloads_dir / f"{video_title}.mp4"
            output_transcript = downloads_dir / f"{video_title}.txt"
            logging.info(f"  Video: {output_video}")
            logging.info(f"  Transcript: {output_transcript}")

    except (ValueError, requests.RequestException, yt_dlp.utils.DownloadError, FileNotFoundError) as e:
        logging.error(f"Process failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()