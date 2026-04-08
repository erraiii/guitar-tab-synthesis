from ultralytics import YOLO
from utils.parsing import parse_guitar_detections


class GuitarDetector:
    def __init__(self, model_path: str, conf: float = 0.2):
        self.model = YOLO(model_path)
        self.conf = conf
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

