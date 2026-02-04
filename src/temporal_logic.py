from collections import deque
import time


class DrowsinessMonitor:
    def __init__(self, ear_threshold=0.25, frames_threshold=20):
        self.ear_threshold = ear_threshold
        self.frames_threshold = frames_threshold

        self.counter = 0
        self.drowsy = False

        self.blink_count = 0
        self.eye_closed = False

    def update(self, ear):
        status = "ALERT"

        if ear < self.ear_threshold:
            self.counter += 1

            if not self.eye_closed:
                self.eye_closed = True
                self.blink_count += 1

            if self.counter >= self.frames_threshold:
                self.drowsy = True
                status = "DROWSY"
        else:
            self.counter = 0
            self.eye_closed = False
            self.drowsy = False

        return status
