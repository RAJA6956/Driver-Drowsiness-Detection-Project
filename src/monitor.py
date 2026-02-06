# src/monitor.py

import numpy as np
from scipy.spatial import distance as dist


class DrowsinessMonitor:
    def __init__(self):
        # ===== EAR (blink / drowsy) =====
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 15

        self.ear_counter = 0
        self.blink_count = 0
        self.drowsy = False

        # ===== MAR (yawning) =====
        self.MAR_THRESHOLD = 0.60      # realistic
        self.YAWN_FRAMES = 10           # ~0.3 sec

        self.mar_counter = 0
        self.yawn_count = 0
        self.yawning = False
        self._yawn_latched = False     # CRITICAL FIX

    # -------------------------------
    # EAR computation
    # -------------------------------
    def compute_ear(self, landmarks):
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        def eye_aspect_ratio(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        return (left_ear + right_ear) / 2.0

    # -------------------------------
    # MAR computation
    # -------------------------------
    def compute_mar(self, landmarks):
        mouth = landmarks[48:68]

        A = dist.euclidean(mouth[13], mouth[19])  # upper-lower
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])  # width

        return (A + B + C) / (3.0 * D)

    # -------------------------------
    # UPDATE LOGIC
    # -------------------------------
    def update(self, ear, mar):
        # ===== EAR LOGIC =====
        if ear < self.EAR_THRESHOLD:
            self.ear_counter += 1
            if self.ear_counter >= self.EAR_CONSEC_FRAMES:
                self.drowsy = True
        else:
            if self.ear_counter >= self.EAR_CONSEC_FRAMES:
                self.blink_count += 1
            self.ear_counter = 0
            self.drowsy = False

        # ===== MAR (YAWN) LOGIC — FIXED =====
        if mar > self.MAR_THRESHOLD:
            self.mar_counter += 1
            if self.mar_counter >= self.YAWN_FRAMES:
                self.yawning = True

                # increment ONLY ONCE per yawn
                if not self._yawn_latched:
                    self.yawn_count += 1
                    self._yawn_latched = True
        else:
            self.mar_counter = 0
            self.yawning = False
            self._yawn_latched = False

        return "DROWSY" if self.drowsy else "ALERT"
