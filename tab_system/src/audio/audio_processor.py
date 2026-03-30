import requests
from core.models import AudioNote, Note


class AudioProcessor:
    def __init__(self):
        print("[AudioProcessor] init")

    def process(self, audio_path: str):
        print(f"[AudioProcessor] process {audio_path}")

        # отправляем аудио в контейнер
        with open(audio_path, "rb") as f:
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": f}
            )

        # получаем JSON
        data = response.json()

        audio_notes = []

        # преобразуем JSON в AudioNote
        for note in data["notes"]:

            # фильтр по уверенности
            if note["confidence"] < 0.2:
                continue

            audio_notes.append(
                AudioNote(
                    start=note["start"],
                    end=note["end"],
                    notes=[Note(midi=note["pitch"])]
                )
            )

        print(f"[AudioProcessor] got {len(audio_notes)} notes")

        return audio_notes