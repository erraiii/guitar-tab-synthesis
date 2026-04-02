from config import VIDEO_PATH
from fusion.tab_generator import TabGenerator
from audio.audio_processor import AudioProcessor
from visual.visual_processor import VisualProcessor
from utils.visualization import show_frame
from utils.audio import delete_audio


if __name__ == "__main__":
    vp = VisualProcessor(VIDEO_PATH)

    a_path = vp.extract_audio()

    ap = AudioProcessor()
    try:
        notes = ap.process(a_path)
    finally:
        delete_audio(a_path)

    # print(notes[:5])
    for note in notes:
        frame = vp.get_frame_at(note.start)
        show_frame(frame)

    vp.release()
