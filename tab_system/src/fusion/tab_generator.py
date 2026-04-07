from audio.audio_processor import AudioProcessor
from config import MODEL_PATH
from fusion.fingering_processor import FingeringProcessor
from geometry.primitives import remove_duplicate_frets
from utils.audio import delete_audio
from visual.hand_detection import HandDetector, HandTracker, get_closest_hand
from visual.visual_processor import VisualProcessor
from visual.guitar_detector import GuitarDetector
from geometry.geometry_processor import GeometryProcessor
from utils.visualization import show_frame, draw_hands, visualize_detections, draw_midstrings


class TabGenerator:
    def __init__(self, video_path: str):
        self.audio_processor = AudioProcessor()
        self.visual_processor = VisualProcessor(video_path)
        self.hand_detector = HandDetector()
        self.guitar_detector = GuitarDetector(MODEL_PATH)
        self.geometry_processor = GeometryProcessor()
        self.fingering_processor = FingeringProcessor()

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
            guitar.frets = remove_duplicate_frets(guitar.frets)

            # fallback
            if guitar is None or len(guitar.frets) == 0:
                guitar = prev_guitar

            prev_guitar = guitar

            # --GEOMETRY--
            if guitar is not None and len(guitar.frets) > 0:
                # строим линии струн через GeometryProcessor
                midstrings_abc, fret_lines = self.geometry_processor.process(hand['box'], guitar, frame.shape)

                fingering = self.fingering_processor.detect(
                    hand["fingertips"],
                    fret_lines,
                    midstrings_abc,
                    t
                )
                print(fingering)
                # рисуем межструнные линии
                frame = draw_midstrings(frame, midstrings_abc)
                frame = draw_midstrings(frame, fret_lines)

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

