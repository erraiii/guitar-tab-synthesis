from fusion.tab_generator import TabGenerator
from audio.audio_processor import AudioProcessor
from visual.visual_processor import VisualProcessor
from utils.visualization import show_frame


if __name__ == "__main__":
    vp = VisualProcessor("path")
    frame = vp.get_frame_at(17.0)
    show_frame(frame)
    vp.release()

    ap = AudioProcessor()
    notes = ap.process("audio_path")

    print(notes[:5])