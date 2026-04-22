from ultralytics import YOLO
from config import GUITAR_DETECT_CONFIDENCE
from utils.parsing import parse_guitar_detections
import logging

logger = logging.getLogger(__name__)


class GuitarDetector:
    def __init__(self, model_path: str, conf: float = None):
        logger.debug(f"Initializing GuitarDetector with model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf if conf is not None else GUITAR_DETECT_CONFIDENCE
        self.class_names = self.model.names

    def detect(self, image, time=None):
        logger.debug(f"Detecting guitar at time {time}")
        results = self.model.track(
            image,
            conf=self.conf,
            persist=True,
            verbose=False
        )

        if not results:
            logger.debug("No detection results")
            return None

        res = results[0]

        return parse_guitar_detections(
            res,
            self.class_names,
            time
        )

