from fusion.tab_generator import TabGenerator
from audio.audio_processor import AudioProcessor


if __name__ == "__main__":
    ap = AudioProcessor()
    notes = ap.process("path")

    print(notes[:5])
