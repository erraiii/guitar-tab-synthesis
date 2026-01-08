import cv2
import mediapipe as mp


class HandLandmarkTracker:
    """
    Детекция и трекинг ключевых точек кистей рук
    с использованием MediaPipe Hands.
    """

    def __init__(self,
                 max_hands=2,
                 detection_conf=0.5,
                 tracking_conf=0.5):

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def process_frame(self, frame):
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        output = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        detected = []

        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
            ):
                label = handedness.classification[0].label
                score = handedness.classification[0].score

                keypoints = [
                    (i, int(lm.x * w), int(lm.y * h))
                    for i, lm in enumerate(landmarks.landmark)
                ]

                detected.append({
                    'label': label,
                    'confidence': float(score),
                    'keypoints': keypoints,
                    'landmarks_raw': landmarks
                })

        # максимум 1 левая и 1 правая
        hands_filtered = []
        for side in ('Left', 'Right'):
            candidates = [h for h in detected if h['label'] == side]
            if candidates:
                best = max(candidates, key=lambda x: x['confidence'])
                hands_filtered.append(best)

        for i, hand in enumerate(hands_filtered):
            self.mp_drawing.draw_landmarks(
                output,
                hand['landmarks_raw'],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

            cv2.putText(
                output,
                f"{hand['label']} ({hand['confidence']:.2f})",
                (10, 30 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        hands_data = [
            {
                'label': h['label'],
                'confidence': h['confidence'],
                'keypoints': h['keypoints']
            }
            for h in hands_filtered
        ]

        return output, hands_data

    def run(self, camera_id=0):
        """
        Запуск трекинга с веб-камеры.
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Hand tracking запущен. Нажмите 'q' для выхода.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            vis, hands = self.process_frame(frame)

            cv2.putText(
                vis,
                f'Hands: {len(hands)}',
                (10, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow('HandLandmarkTracker', vis)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()



if __name__ == "__main__":
    tracker = HandLandmarkTracker()
    tracker.run()
