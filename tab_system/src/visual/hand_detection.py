from config import HAND_MODEL_PATH, HAND_MODEL_URL, MAX_HANDS, HAND_TRACKING_STEP, HAND_BOX_PADDING
from utils.download import download_if_missing
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


class HandDetector:
    def __init__(self):
        download_if_missing(HAND_MODEL_PATH, HAND_MODEL_URL)

        base_options = python.BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, timestamp):
        frame_rgb = frame[:, :, ::-1]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = self.detector.detect_for_video(mp_image, timestamp)

        return result


class HandTracker:
    def __init__(self, visual_processor, step=0.1):
        self.vp = visual_processor
        self.detector = HandDetector()
        self.step = step

    def track(self, duration):
        results = []

        frame_idx = 0
        frame_step = HAND_TRACKING_STEP  # каждый N-й кадр

        total_frames = int(self.vp.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_idx < total_frames:
            t = frame_idx / self.vp.fps

            frame = self.vp.get_frame_at(t)

            if frame is None:
                frame_idx += frame_step
                continue

            result = self.detector.detect(frame, int(t * 1000))

            if result.hand_landmarks:
                best_hand = None
                best_score = -1

                for i, hand in enumerate(result.hand_landmarks):
                    handedness = result.handedness[i][0]

                    if handedness.category_name != "Left":
                        continue

                    score = handedness.score

                    if score > best_score:
                        best_score = score
                        best_hand = hand

                if best_hand is not None:
                    box, _ = landmarks_to_boxes([best_hand], frame.shape)
                    fingertips = extract_fingertips(best_hand, frame.shape)

                    results.append({
                        "time": t,
                        "box": box[0],
                        "fingertips": fingertips
                    })

            frame_idx += frame_step

        return results


def extract_fingertips(hand_landmarks, img_shape):
    h, w = img_shape[:2]

    fingertip_ids = [8, 12, 16, 20]  # большой - 4

    fingertips = []

    for lm_id in fingertip_ids:
        lm = hand_landmarks[lm_id]
        x = int(lm.x * w)
        y = int(lm.y * h)
        fingertips.append((x, y))

    return fingertips


def landmarks_to_boxes(hand_landmarks, img_shape, padding=None):
    """
    Преобразует landmarks в bounding boxes
    """
    if padding is None:
        padding = HAND_BOX_PADDING

    h, w = img_shape[:2]

    hand_boxes = []
    landmarks_list = []

    for hand in hand_landmarks:

        xs = [lm.x * w for lm in hand]
        ys = [lm.y * h for lm in hand]

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        pad_x = int(padding * (x2 - x1))
        pad_y = int(padding * (y2 - y1))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        hand_boxes.append([x1, y1, x2, y2])
        landmarks_list.append(hand)

    return hand_boxes, landmarks_list


def get_closest_hand(hand_data, note_time):
    if not hand_data:
        return None

    return min(hand_data, key=lambda x: abs(x["time"] - note_time))