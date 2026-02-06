# src/monitor.py

class DrowsinessMonitor:
    """
    Tracks EAR/MAR over time and determines driver state.
    """

    def __init__(
        self,
        ear_threshold=0.25,
        mar_threshold=0.6,
        drowsy_frames=20
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.drowsy_frames = drowsy_frames

        self.closed_eye_frames = 0
        self.blink_count = 0
        self.state = "AWAKE"

    def update(self, ear, mar):
        if ear < self.ear_threshold:
            self.closed_eye_frames += 1
        else:
            if self.closed_eye_frames > 2:
                self.blink_count += 1
            self.closed_eye_frames = 0

        if self.closed_eye_frames >= self.drowsy_frames or mar > self.mar_threshold:
            self.state = "DROWSY"
        else:
            self.state = "AWAKE"

        return self.state
