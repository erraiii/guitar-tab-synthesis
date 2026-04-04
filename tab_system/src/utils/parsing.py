import numpy as np
from core.models import Fret, Capo, Nut, GuitarDetections


def get_obb_boxes(results):
    boxes = []

    obb = results.obb
    if obb is None:
        return boxes

    xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
    classes = obb.cls.cpu().numpy().astype(int)

    for i in range(len(xyxyxyxy)):
        corners = xyxyxyxy[i]
        class_id = classes[i]

        boxes.append((class_id, corners))

    return boxes


def parse_guitar_detections(res, class_names, time=None):
    boxes = get_obb_boxes(res)

    frets = []
    nut = None
    capo = None

    for class_id, corners in boxes:
        class_name = class_names[class_id]
        corners = np.asarray(corners).reshape(4, 2)

        if class_name == "fret":
            frets.append(Fret(corners=corners))

        elif class_name == "nut":
            nut = Nut(corners=corners)

        elif class_name == "capo":
            capo = Capo(corners=corners)

    return GuitarDetections(
        frets=frets,
        nut=nut,
        capo=capo,
        time=time
    )