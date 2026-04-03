import matplotlib.pyplot as plt
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