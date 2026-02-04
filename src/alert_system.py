import time
import winsound


class AlertSystem:
    def __init__(self, cooldown=2):
        self.last_alert_time = 0
        self.cooldown = cooldown

    def trigger_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time >= self.cooldown:
            winsound.Beep(2500, 800)  # frequency, duration
            self.last_alert_time = current_time
