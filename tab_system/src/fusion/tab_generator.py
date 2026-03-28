from audio.audio_processor import AudioProcessor
from visual.visual_processor import VisualProcessor


class TabGenerator:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.visual_processor = VisualProcessor()

    def generate(self, video_path: str):
        print(f"[TabGenerator] Generating tabs for {video_path}")

        audio_path = self.extract_audio(video_path)
        audio_notes = self.audio_processor.process(audio_path)

        for note in audio_notes:
            frame = self.get_frame(video_path, note.timestamp)
            visual = self.visual_processor.process(frame)

            print(note, visual)

    def extract_audio(self, video_path: str):
        print("[TabGenerator] extract audio")
        return "temp.wav"

    def get_frame(self, video_path: str, timestamp: float):
        print(f"[TabGenerator] get frame at {timestamp}")
        return None