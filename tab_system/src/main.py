from fusion.tab_generator import TabGenerator
from audio.audio_processor import AudioProcessor


if __name__ == "__main__":
    ap = AudioProcessor()
    notes = ap.process("C:/Users/aivan/Downloads/4-мар._-17.24_.wav")

    print(notes[:5])
