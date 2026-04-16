from collections import Counter
from audio.audio_processor import AudioProcessor
from config import MODEL_PATH
from core.tab_builder import TabBuilder, save_tabs_pdf
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

        audio_events = self._extract_audio_events()
        hand_data = self._track_hands()
        frames_data = self._build_frames_data(audio_events, hand_data)

        final_capo = self._resolve_capo(frames_data)
        self._fuse_audio_visual(frames_data, final_capo)

        tabs_content = self._render_tabs(frames_data, final_capo)
        self.visual_processor.release()

        return tabs_content

    def _extract_audio_events(self):
        print("[TabGenerator] extracting audio events")
        audio_path = self.visual_processor.extract_audio()

        try:
            return self.audio_processor.process(audio_path)
        finally:
            print("[TabGenerator] deleting audio")
            delete_audio(audio_path)

    def _track_hands(self):
        print("[TabGenerator] tracking hands")
        tracker = HandTracker(self.visual_processor)
        return tracker.track(self.visual_processor.duration)

    def _build_frames_data(self, audio_events, hand_data):
        frames_data = []
        prev_guitar = None

        for event in audio_events:
            try:
                frame_data = self._process_event(event, hand_data, prev_guitar)
                if frame_data is None:
                    continue

                prev_guitar = frame_data["guitar"] or prev_guitar
                frames_data.append(frame_data)
            except Exception as e:
                print(f"[TabGenerator] skip event at {event.start:.3f}s: {e}")
                continue

        return frames_data

    def _process_event(self, event, hand_data, prev_guitar):
        t = event.start
        raw_frame = self.visual_processor.get_frame_at(t)
        if raw_frame is None:
            return None

        hand = get_closest_hand(hand_data, t)
        # frame = raw_frame.copy()
        # if hand is not None:
        #     frame = draw_hands(frame, hand["box"], hand["fingertips"])

        guitar = self.guitar_detector.detect(raw_frame, time=t)
        if guitar is not None:
            guitar.frets = remove_duplicate_frets(guitar.frets)

        if guitar is None or len(guitar.frets) == 0:
            guitar = prev_guitar

        if guitar is not None and len(guitar.frets) > 0:
            geometry_result = self.geometry_processor.process(
                hand["box"] if hand else None,
                guitar,
                raw_frame.shape
            )

            if geometry_result is None:
                print(f"[TabGenerator] geometry processing failed at {t:.3f}s")
                string_lines = []
                fret_lines = []
            else:
                string_lines, fret_lines = geometry_result

            fingering = None
            if hand is not None and fret_lines and string_lines:
                try:
                    fingering = self.fingering_processor.detect(
                        hand["fingertips"],
                        fret_lines,
                        string_lines,
                        t
                    )
                except Exception as e:
                    print(f"[TabGenerator] fingering detection failed at {t:.3f}s: {e}")
                    fingering = None

            capo_fret = None
            if guitar.capo is not None and fret_lines and string_lines:
                try:
                    capo_region = point_to_region(
                        guitar.capo.center,
                        fret_lines,
                        string_lines
                    )
                    if capo_region is not None:
                        _, capo_fret = capo_region
                except Exception as e:
                    print(f"[TabGenerator] capo detection failed at {t:.3f}s: {e}")
                    capo_fret = None
        else:
            string_lines = []
            fret_lines = []
            fingering = None
            capo_fret = None

        return {
            "note": event,
            "guitar": guitar,
            "hand": hand,
            "fingering": fingering,
            "string_lines": string_lines,
            "fret_lines": fret_lines,
            "capo_fret": capo_fret,
        }

    def _resolve_capo(self, frames_data):
        capo_values = [item["capo_fret"] for item in frames_data if item["capo_fret"] is not None]
        if not capo_values:
            return None
        return Counter(capo_values).most_common(1)[0][0]

    def _fuse_audio_visual(self, frames_data, final_capo):
        for item in frames_data:
            touch_positions = item["fingering"].positions if item["fingering"] is not None else []
            visual_candidates = generate_visual_candidates(
                touch_positions,
                capo=final_capo
            )

            item["fused"] = self.fusion_processor.fuse_event(
                item["note"],
                item["fingering"],
                visual_candidates
            )

    def _render_tabs(self, frames_data, final_capo):
        builder = TabBuilder(capo=final_capo)
        for item in frames_data:
            builder.add_event(item.get("fused", []))

        tabs_content = builder.render_chunked()

        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "tabs.txt"
        output_pdf_path = output_dir / "tabs.pdf"

        output_path.write_text(tabs_content, encoding="utf-8")
        save_tabs_pdf(tabs_content, output_pdf_path)

        print(f"[TabGenerator] Tabs saved to {output_path}")
        print(f"[TabGenerator] Tabs saved to {output_pdf_path}")

        return tabs_content

