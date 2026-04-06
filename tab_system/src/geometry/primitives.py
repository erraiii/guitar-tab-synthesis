import numpy as np


def remove_duplicate_frets(frets, iou_threshold=0.9):

    def iou(boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union = areaA + areaB - inter

        return inter / union if union > 0 else 0

    def center_inside(box, bbox):
        """проверяем, лежит ли центр box внутри bbox"""
        box = [float(x) for x in box]
        bbox = [float(x) for x in bbox]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]

    unique = []

    for f in frets:
        bboxA = polygon_to_bbox(f.corners)
        duplicate = False

        for u in unique:
            bboxB = polygon_to_bbox(u.corners)
            # стандартный IoU
            if iou(bboxA, bboxB) > iou_threshold:
                duplicate = True
                break
            # центр бокса внутри другого
            if center_inside(bboxA, bboxB) or center_inside(bboxB, bboxA):
                duplicate = True
                break

        if not duplicate:
            unique.append(f)

    return unique



def filter_by_hands(frets, hand_boxes):
    if hand_boxes and not isinstance(hand_boxes[0], (list, tuple)):
        hand_boxes = [hand_boxes]

    filtered = []
    rejected = []

    for fret in frets:

        fret_bbox = polygon_to_bbox(fret.corners)

        intersects = False

        for hand_box in hand_boxes:
            # 1. стандартное пересечение
            if bboxes_intersect(fret_bbox, hand_box):
                intersects = True
                break

            # 2. проверка по горизонтали (ось X)
            overlap_x = max(0, min(fret_bbox[2], hand_box[2]) - max(fret_bbox[0], hand_box[0]))
            fret_width = fret_bbox[2] - fret_bbox[0]
            if overlap_x / fret_width > 0.2:
                intersects = True
                break

        if intersects:
            rejected.append(fret)
        else:
            filtered.append(fret)

    return filtered, rejected


def align_string_direction(frets):

    if not frets:
        return frets

    ref_p1, ref_p2 = frets[0].center_line
    ref_dir = ref_p2 - ref_p1

    for f in frets:
        p1, p2 = f.center_line
        cur_dir = p2 - p1

        if np.dot(ref_dir, cur_dir) < 0:
            f.center_line = (p2, p1)

    return frets


def compute_line_pts(line_pts, outer_gap_ratio=0.18):
    """
    line_pts: np.array([[x1,y1],[x2,y2]])

    returns:
        (7,2) numpy array
    """

    p1, p2 = line_pts

    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return None

    direction = direction / length

    center = (p1 + p2) / 2

    outer_gap = outer_gap_ratio * length
    span = length - 2 * outer_gap

    offsets = np.linspace(
        -span/2 - outer_gap,
         span/2 + outer_gap,
         7
    )

    pts = np.array([
        center + off * direction
        for off in offsets
    ])

    return pts


def compute_mean_direction(frets):
    dirs = []

    for f in frets:
        p1, p2 = f.center_line
        d = p2 - p1

        norm = np.linalg.norm(d)
        if norm == 0:
            continue

        dirs.append(d / norm)

    if not dirs:
        return None

    mean_dir = np.mean(dirs, axis=0)

    norm = np.linalg.norm(mean_dir)
    if norm == 0:
        return None

    return mean_dir / norm


def fit_line(points):
    """
    Построение линии через SVD.

    points: np.array shape (N,2)

    return:
        centroid: np.array(2,) — точка на линии
        direction: np.array(2,) — нормированный вектор вдоль линии
        abc: tuple (a,b,c) — коэффициенты линии в формате ax + by + c = 0
    """
    pts = np.array(points)
    if len(pts) < 2:
        return None, None, None  # мало точек для линии

    # Центроид
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # SVD
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]
    direction = direction / np.linalg.norm(direction)

    # Преобразование в формат ax + by + c = 0
    a = -direction[1]
    b = direction[0]
    c = -(a * centroid[0] + b * centroid[1])

    return (a, b, c) # centroid, direction

# -- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ --
def polygon_to_bbox(corners):
    """
    corners: (4,2)
    returns: [x1, y1, x2, y2]
    """
    xs = corners[:, 0]
    ys = corners[:, 1]

    return [xs.min(), ys.min(), xs.max(), ys.max()]


def bboxes_intersect(boxA, boxB):
    # boxA = [float(x) for x in boxA]
    # boxB = [float(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    return (xA < xB) and (yA < yB)