import csv
import os
from datetime import datetime


class SessionLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(log_dir, f"session_{timestamp}.csv")

        self.file = open(self.file_path, mode="w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "time",
            "EAR",
            "MAR",
            "blink_count",
            "drowsy"
        ])

    def log(self, ear, mar, blink_count, drowsy):
        self.writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            round(ear, 3),
            round(mar, 3),
            blink_count,
            drowsy
        ])

    def close(self):
        self.file.close()
