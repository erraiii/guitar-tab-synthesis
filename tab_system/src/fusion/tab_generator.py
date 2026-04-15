from collections import Counter
from audio.audio_processor import AudioProcessor
from config import MODEL_PATH
from core.tab_builder import TabBuilder
from fusion.fingering_processor import FingeringProcessor
from geometry.primitives import remove_duplicate_frets
from geometry.region import point_to_region
from utils.audio import delete_audio
from visual.hand_detection import HandTracker, get_closest_hand
from visual.visual_processor import VisualProcessor
from visual.guitar_detector import GuitarDetector
from geometry.geometry_processor import GeometryProcessor
from utils.visualization import draw_hands
from fusion.fusion_processor import FusionProcessor
from fusion.fret_mapper import FretboardMapper
from fusion.candidates import generate_visual_candidates
from config import PROJECT_ROOT


class TabGenerator:
    def __init__(self, video_path: str):
        self.audio_processor = AudioProcessor()
        self.visual_processor = VisualProcessor(video_path)
        self.guitar_detector = GuitarDetector(MODEL_PATH)
        self.geometry_processor = GeometryProcessor()
        self.fingering_processor = FingeringProcessor()
        self.mapper = FretboardMapper()
        self.fusion_processor = FusionProcessor(self.mapper)

    def generate(self):
        print(f"[TabGenerator] Generating tabs")

        # --AUDIO--
        print("[VisualProcessor] extract audio")
        audio_path = self.visual_processor.extract_audio()
        try:
            audio_events = self.audio_processor.process(audio_path)
        finally:
            print("[TabGenerator] delete audio")
            delete_audio(audio_path)

        # --HANDS--
        tracker = HandTracker(self.visual_processor)
        hand_data = tracker.track(self.visual_processor.duration)

        # --MAIN LOOP--
        prev_guitar = None
        capo_history = []
        frames_data = []
        for event in audio_events:
            t = event.start
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
                # строим линии струн через GeometryProcessor (рука может быть None)
                midstrings_abc, fret_lines = self.geometry_processor.process(hand['box'] if hand else None, guitar, frame.shape)

                if hand is not None:
                    fingering = self.fingering_processor.detect(
                        hand["fingertips"],
                        fret_lines,
                        midstrings_abc,
                        t
                    )
                    print(fingering)
                else:
                    fingering = None
            else:
                # гитара не обнаружена — ничего не строим
                fingering = None
                fret_lines = []
                midstrings_abc = []

            capo_fret = None

            if guitar is not None and guitar.capo is not None:
                capo_center = guitar.capo.center

                _, capo_fret = point_to_region(
                    capo_center,
                    fret_lines,
                    midstrings_abc
                )

            capo_history.append(capo_fret)

            frames_data.append({
                "note": event,
                "fingering": fingering,
                "fret_lines": fret_lines,
                "midstrings": midstrings_abc,
            })

        valid = [c for c in capo_history if c is not None]

        if valid:
            final_capo = Counter(valid).most_common(1)[0][0]
        else:
            final_capo = None

        for data in frames_data:
            note = data["note"]
            fingering = data["fingering"]

            # --- визуальные кандидаты ---
            visual_candidates = generate_visual_candidates(
                fingering.positions,
                capo=final_capo
            )

            # --- fusion ---
            fused = self.fusion_processor.fuse_event(
                note,
                fingering,
                visual_candidates
            )

            data["fused"] = fused

        tab_builder = TabBuilder(capo=final_capo)

        for data in frames_data:
            tab_builder.add_event(data["fused"])

        # Write tabs to output file
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "tabs.txt"
        tabs_content = tab_builder.render_chunked()
        output_path.write_text(tabs_content, encoding="utf-8")
        print(f"[TabGenerator] Tabs saved to {output_path}")

        self.visual_processor.release()

