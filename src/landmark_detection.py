# src/landmark_detection.py

import cv2
import mediapipe as mp


class LandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Indices for eyes and mouth (MediaPipe standard)
        self.eye_indices = [
            33, 133, 160, 159, 158, 144,   # left eye
            362, 263, 387, 386, 385, 373   # right eye
        ]

        self.mouth_indices = [
            61, 291, 81, 178, 13, 14
        ]

    def detect(self, frame, face_rect=None):
        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        points = {}
        for idx in self.eye_indices + self.mouth_indices:
            lm = landmarks[idx]
            points[idx] = (int(lm.x * w), int(lm.y * h))

        return points

    def draw(self, frame, landmarks):
        for (x, y) in landmarks.values():
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
