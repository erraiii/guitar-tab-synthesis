from pathlib import Path
import urllib.request


def download_if_missing(path: Path, url: str):

    path = Path(path)

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model to {path}")

    urllib.request.urlretrieve(url, path)

