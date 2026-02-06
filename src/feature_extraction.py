# src/feature_extraction.py

import math


class FeatureExtractor:
    """
    Computes EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio)
    from MediaPipe facial landmarks.
    """

    def __init__(self):
        # Landmark indices based on MediaPipe FaceMesh
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]
        self.mouth = [61, 291, 81, 178, 13, 14]

    def _euclidean(self, p1, p2):
        return math.dist(p1, p2)

    def compute_ear(self, landmarks):
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        def eye_ratio(indices):
            p = [landmarks[i] for i in indices]
            vertical1 = self._euclidean(p[1], p[5])
            vertical2 = self._euclidean(p[2], p[4])
            horizontal = self._euclidean(p[0], p[3])
            return (vertical1 + vertical2) / (2.0 * horizontal)

        left_ear = eye_ratio(self.left_eye)
        right_ear = eye_ratio(self.right_eye)

        return (left_ear + right_ear) / 2.0

    def compute_mar(self, landmarks):
        """
        MAR = ||p2-p6|| / ||p1-p4||
        """
        p = [landmarks[i] for i in self.mouth]
        vertical = self._euclidean(p[2], p[3])
        horizontal = self._euclidean(p[0], p[1])
        return vertical / horizontal
