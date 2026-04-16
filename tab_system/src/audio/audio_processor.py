import requests
from requests.exceptions import RequestException, Timeout
from config import AUDIO_SERVICE_URL, AUDIO_SERVICE_TIMEOUT, AUDIO_CONFIDENCE_THRESHOLD
from core.models import AudioNote
from audio.postprocessing import process_notes


class AudioProcessor:
    def __init__(self):
        print("[AudioProcessor] init")

    def process(self, audio_path: str):
        print(f"[AudioProcessor] process {audio_path}")

        try:
            with open(audio_path, "rb") as f:
                response = requests.post(
                    AUDIO_SERVICE_URL,
                    files={"file": f},
                    timeout=AUDIO_SERVICE_TIMEOUT
                )

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to audio service at {AUDIO_SERVICE_URL}. "
                f"Is the server running?"
            ) from None
        except Timeout:
            raise RuntimeError(
                f"Audio service at {AUDIO_SERVICE_URL} timed out "
                f"after {AUDIO_SERVICE_TIMEOUT}s"
            )
        except RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with audio service: {e}"
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Audio service returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as e:
            raise RuntimeError(
                f"Audio service returned invalid JSON: {e}"
            )

        if "notes" not in data:
            raise RuntimeError(
                f"Unexpected audio service response: missing 'notes' field"
            )

        audio_notes = []

        for note in data["notes"]:
            if note.get("confidence", 0) < AUDIO_CONFIDENCE_THRESHOLD:
                continue

            audio_notes.append(
                AudioNote(
                    start=note["start"],
                    end=note["end"],
                    pitch=note["pitch"]
                )
            )

        print(f"[AudioProcessor] got {len(audio_notes)} notes")
        audio_notes = process_notes(audio_notes)
        print(f"[AudioProcessor] got {len(audio_notes)} notes after filters")

        return audio_notes