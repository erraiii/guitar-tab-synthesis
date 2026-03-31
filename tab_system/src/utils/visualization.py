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