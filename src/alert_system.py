# src/alert_system.py

import threading
import time

try:
    from playsound import playsound
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False


class AlertSystem:
    def __init__(self, sound_path="alert.wav", cooldown=3.0):
        self.sound_path = sound_path
        self.cooldown = cooldown
        self.last_played = 0.0
        self._lock = threading.Lock()

    def play(self):
        if not AUDIO_AVAILABLE:
            return

        now = time.time()

        with self._lock:
            if now - self.last_played < self.cooldown:
                return
            self.last_played = now

        t = threading.Thread(target=self._play_sound, daemon=True)
        t.start()

    def _play_sound(self):
        try:
            playsound(self.sound_path)
        except Exception:
            pass
