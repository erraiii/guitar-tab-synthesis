from ultralytics import YOLO
from config import GUITAR_DETECT_CONFIDENCE
from utils.parsing import parse_guitar_detections


class GuitarDetector:
    def __init__(self, model_path: str, conf: float = None):
        self.model = YOLO(model_path)
        self.conf = conf if conf is not None else GUITAR_DETECT_CONFIDENCE
        self.class_names = self.model.names

    def detect(self, image, time=None):
        results = self.model.track(
            image,
            conf=self.conf,
            persist=True,
            verbose=False
        )

        if not results:
            return None

        res = results[0]

        return parse_guitar_detections(
            res,
            self.class_names,
            time
        )

