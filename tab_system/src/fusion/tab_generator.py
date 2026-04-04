from audio.audio_processor import AudioProcessor
from config import MODEL_PATH
from utils.audio import delete_audio
from visual.hand_detection import HandDetector, HandTracker, get_closest_hand
from visual.visual_processor import VisualProcessor
from visual.guitar_detector import GuitarDetectionPipeline, GuitarDetector
from utils.visualization import show_frame, draw_hands, visualize_detections


class TabGenerator:
    def __init__(self, video_path: str):
        self.audio_processor = AudioProcessor()
        self.visual_processor = VisualProcessor(video_path)
        self.hand_detector = HandDetector()
        self.guitar_detector = GuitarDetector(MODEL_PATH)
        '''
        self.guitar_pipeline = GuitarDetectionPipeline(
            self.visual_processor,
            self.fret_detector
        )'''

    def generate(self):
        print(f"[TabGenerator] Generating tabs")

        # --AUDIO--
        print("[VisualProcessor] extract audio")
        audio_path = self.visual_processor.extract_audio()
        try:
            audio_notes = self.audio_processor.process(audio_path)
        finally:
            print("[TabGenerator] delete audio")
            delete_audio(audio_path)

        # --HANDS--
        tracker = HandTracker(self.visual_processor)
        hand_data = tracker.track(self.visual_processor.duration)

        # --MAIN LOOP--
        # fret_results = self.guitar_pipeline.process_notes(audio_notes)
        # print(fret_results)
        # print(int(input()))
        prev_guitar = None

        for note in audio_notes:
            t = note.start
            raw_frame = self.visual_processor.get_frame_at(t)
            if raw_frame is None:
                continue

            frame = raw_frame.copy()
            # --HAND--
            hand = get_closest_hand(hand_data, t)
            if hand is not None:
                frame = draw_hands(frame, hand["box"], hand["fingertips"])

            # --GUITAR--
            guitar = self.guitar_detector.detect(raw_frame, time=t)

            # fallback
            if guitar is None or len(guitar.frets) == 0:
                guitar = prev_guitar

            prev_guitar = guitar

            # --VISUALIZE GUITAR--
            frame = visualize_detections(
                frame,
                guitar,
                show=False,
                return_img=True
            )
            # --SHOW--
            show_frame(frame)

        self.visual_processor.release()

