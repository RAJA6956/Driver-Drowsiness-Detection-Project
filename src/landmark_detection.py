# src/landmark_detection.py

import cv2
import mediapipe as mp
import numpy as np


class LandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [61, 291, 81, 178, 13, 14]

    def _normalize_rect(self, rect):
        """
        Accepts either:
        (x, y, w, h) OR (left, top, right, bottom)
        Returns: x, y, w, h
        """
        x1, y1, x2, y2 = rect

        # If right/bottom are larger than left/top → assume (l, t, r, b)
        if x2 > x1 and y2 > y1:
            return x1, y1, x2 - x1, y2 - y1

        # Otherwise assume already (x, y, w, h)
        return rect

    def detect(self, frame, face_rect):
        try:
            x, y, w, h = self._normalize_rect(face_rect)

            h_frame, w_frame = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)

            if w <= 0 or h <= 0:
                return None

            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size == 0:
                return None

            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                return None

            landmarks = result.multi_face_landmarks[0]

            points = []
            for lm in landmarks.landmark:
                px = int(lm.x * w) + x
                py = int(lm.y * h) + y
                points.append((px, py))

            return points

        except Exception:
            return None

    def draw(self, frame, landmarks):
        if landmarks is None:
            return

        for idx in self.LEFT_EYE + self.RIGHT_EYE + self.MOUTH:
            if idx < len(landmarks):
                cv2.circle(frame, landmarks[idx], 2, (0, 255, 0), -1)
