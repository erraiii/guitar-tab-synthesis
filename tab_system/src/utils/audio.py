
import subprocess
import os


def extract_audio(video_path: str, output_path: str = None):
    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + ".wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",  # без видео
        "-acodec", "pcm_s16le",  # wav
        "-ar", "44100",
        "-ac", "1",  # моно
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path


def delete_audio(path: str):
    if path and os.path.exists(path):
        try:
            os.remove(path)
            print(f"[delete_audio] removed {path}")
        except Exception as e:
            print(f"[delete_audio] error: {e}")
