from ultralytics import YOLO
from utils.parsing import parse_guitar_detections


class GuitarDetector:
    def __init__(self, model_path: str, conf: float = 0.15):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names

    def detect(self, image, time=None):
        results = self.model.predict(
            image,
            conf=self.conf,
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


class GuitarDetectionPipeline:
    def __init__(self, visual_processor, detector):
        self.vp = visual_processor
        self.detector = detector

    def process_notes(self, notes):
        """
        Для каждой ноты достаёт кадр и делает детекцию грифа
        """
        results = []
        prev_fret = None

        for note in notes:
            t = note.start

            frame = self.vp.get_frame_at(t)

            if frame is None:
                results.append(prev_fret)
                continue

            fret = self.detector.detect(frame, time=t)

            # fallback если YOLO не сработал
            if fret is None or len(fret.frets) == 0:
                fret = prev_fret

            prev_fret = fret
            results.append(fret)

        return results