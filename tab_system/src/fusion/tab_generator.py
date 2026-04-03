from audio.audio_processor import AudioProcessor
from utils.audio import delete_audio
from visual.hand_detection import HandDetector, HandTracker, get_closest_hand
from visual.visual_processor import VisualProcessor
from utils.visualization import show_frame, draw_hands


class TabGenerator:
    def __init__(self, video_path: str):
        self.audio_processor = AudioProcessor()
        self.visual_processor = VisualProcessor(video_path)
        self.hand_detector = HandDetector()

    def generate(self):
        print(f"[TabGenerator] Generating tabs")

        print("[VisualProcessor] extract audio")
        audio_path = self.visual_processor.extract_audio()
        try:
            audio_notes = self.audio_processor.process(audio_path)
        finally:
            print("[TabGenerator] delete audio")
            delete_audio(audio_path)

        tracker = HandTracker(self.visual_processor)

        hand_data = tracker.track(self.visual_processor.duration)
        print(hand_data)
        print(len(hand_data))

        for note in audio_notes:
            frame = self.visual_processor.get_frame_at(note.start)
            hand = get_closest_hand(hand_data, note.start)
            frame = draw_hands(frame, hand["box"], hand["fingertips"])
            show_frame(frame)

        self.visual_processor.release()

