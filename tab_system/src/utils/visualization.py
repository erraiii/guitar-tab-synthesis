import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_frame(frame, title: str = "Frame"):
    if frame is None:
        print("Frame is None")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def draw_hands(image, boxes, landmarks=None):
    img = image.copy()

    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    x1, y1, x2, y2 = boxes
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(img[:, :, ::-1])
    plt.axis("off")

    return img


def draw_obb(image, corners, color=(0, 255, 0), thickness=2):
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def visualize_detections(image, guitar_det, show=True, return_img=False):
    """
    Визуализация детекций грифа

    Parameters:
        image: np.ndarray (BGR)
        guitar_det: GuitarDetections
        show: показать через matplotlib
        return_img: вернуть изображение
    """

    if image is None:
        return None

    vis_img = image.copy()

    if guitar_det is None:
        if show:
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.title("No detections")
            plt.axis("off")
            plt.show()
        return vis_img if return_img else None

    # --- Nut ---
    if guitar_det.nut is not None:
        draw_obb(vis_img, guitar_det.nut.corners, (255, 0, 0))

    # --- Capo ---
    if guitar_det.capo is not None:
        draw_obb(vis_img, guitar_det.capo.corners, (255, 255, 0))

    # --- Frets ---
    for i, fret in enumerate(guitar_det.frets):
        draw_obb(vis_img, fret.corners, (0, 0, 255))

        # индекс (если есть)
        # label = str(fret.index) if fret.index is not None else str(i)

        center = fret.center.astype(int)
        '''
        cv2.putText(
            vis_img,
            label,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )'''

    # --- Time ---
    if guitar_det.time is not None:
        cv2.putText(
            vis_img,
            f"t={guitar_det.time:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    if return_img:
        return vis_img