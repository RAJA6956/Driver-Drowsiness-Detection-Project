# src/face_detection.py

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_confidence=0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence
        )

    def detect(self, frame):
        if frame is None:
            return None

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb)

        if not results.detections:
            return None

        h, w, _ = frame.shape
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)

        return (x, y, w_box, h_box)

    def draw(self, frame, rect):
        x, y, w, h = rect
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
