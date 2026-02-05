import numpy as np


class FeatureExtractor:
    def __init__(self):
        # MediaPipe landmark indices
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]
        self.mouth = [13, 14, 78, 308]

    def _eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def extract(self, landmarks):
        # Convert list to numpy array
        landmarks = np.array(landmarks)

        left_eye_pts = landmarks[self.left_eye]
        right_eye_pts = landmarks[self.right_eye]

        left_ear = self._eye_aspect_ratio(left_eye_pts)
        right_ear = self._eye_aspect_ratio(right_eye_pts)
        ear = (left_ear + right_ear) / 2.0

        # Mouth Aspect Ratio (MAR)
        vertical = np.linalg.norm(landmarks[self.mouth[0]] - landmarks[self.mouth[1]])
        horizontal = np.linalg.norm(landmarks[self.mouth[2]] - landmarks[self.mouth[3]])
        mar = vertical / horizontal

        return ear, mar
