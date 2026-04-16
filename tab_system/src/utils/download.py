from pathlib import Path
import urllib.request
import urllib.error
import socket


def download_if_missing(path: Path, url: str):
    """
    Скачивает модель, если её нет локально.
    
    Args:
        path: путь, где должна быть модель
        url: URL для скачивания
        
    Raises:
        RuntimeError: при ошибке скачивания или сетевых проблемах
    """
    path = Path(path)

    if path.exists():
        print(f"[download] model already exists at {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] downloading model to {path} from {url}")

    try:
        urllib.request.urlretrieve(url, path)
        print(f"[download] successfully downloaded to {path}")
    except socket.timeout:
        raise RuntimeError(
            f"Download timed out while fetching {url}. "
            f"Check your internet connection."
        ) from None
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to download model from {url}: {e.reason}. "
            f"Check the URL and your internet connection."
        ) from None
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"HTTP {e.code} error when downloading from {url}: {e.reason}"
        ) from None
    except IOError as e:
        raise RuntimeError(
            f"Failed to save model to {path}: {e}. "
            f"Check disk space and write permissions."
        ) from None
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while downloading model: {type(e).__name__}: {e}"
        ) from None

