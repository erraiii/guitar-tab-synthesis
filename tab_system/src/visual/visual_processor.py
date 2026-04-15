import cv2
from utils.audio import extract_audio


class VisualProcessor:
    def __init__(self, video_path: str):
        print(f"[VisualProcessor] init {video_path}")
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def get_frame_at(self, timestamp: float):
        # защита от выхода за границы
        timestamp = max(0, min(timestamp, self.duration))

        frame_idx = int(timestamp * self.fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            print(f"[VisualProcessor] failed to read frame at {timestamp}")
            return None

        return frame

    def release(self):
        self.cap.release()

    def extract_audio(self, output_path=None):
        return extract_audio(self.video_path, output_path)