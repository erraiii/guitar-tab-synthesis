import cv2
from utils.audio import extract_audio
import logging

logger = logging.getLogger(__name__)


class VisualProcessor:
    def __init__(self, video_path: str):
        logger.debug(f"Initializing VisualProcessor with {video_path}")
        self.video_path = video_path
        
        try:
            self.cap = cv2.VideoCapture(video_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize video capture: {e}") from e

        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open video file: {video_path}. "
                f"Check that the file exists and is in a supported format."
            )

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
            logger.warning(f"Failed to read frame at {timestamp}")
            return None

        return frame

    def release(self):
        logger.debug("Releasing video capture")
        self.cap.release()

    def extract_audio(self, output_path=None):
        logger.debug("Extracting audio from video")
        return extract_audio(self.video_path, output_path)