class TemporalLogic:
    def __init__(self, ear_thresh=0.25, mar_thresh=0.7, consec_frames=20):
        self.ear_thresh = ear_thresh
        self.mar_thresh = mar_thresh
        self.consec_frames = consec_frames

        self.eye_counter = 0
        self.yawn_counter = 0
        self.drowsy = False

    def update(self, ear, mar):
        # Eye closure logic
        if ear < self.ear_thresh:
            self.eye_counter += 1
        else:
            self.eye_counter = 0

        # Yawn logic
        if mar > self.mar_thresh:
            self.yawn_counter += 1
        else:
            self.yawn_counter = 0

        # Final decision
        if self.eye_counter >= self.consec_frames or self.yawn_counter >= self.consec_frames:
            self.drowsy = True
        else:
            self.drowsy = False

        return self.drowsy
