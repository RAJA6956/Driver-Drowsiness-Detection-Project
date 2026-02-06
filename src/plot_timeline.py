# src/plot_timeline.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
LOG_PATH = Path("logs/session_log.csv")

if not LOG_PATH.exists():
    print("❌ session_log.csv not found in logs/")
    print("➡ Run main.py first to generate logs")
    exit()

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(LOG_PATH)
print("✅ CSV loaded")
print("🧾 CSV columns:", list(df.columns))

# =========================
# REQUIRED SIGNALS
# =========================
required = ["timestamp", "ear", "mar"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================
# TIMESTAMP FIX (IMPORTANT)
# =========================
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if df["timestamp"].isna().any():
    raise ValueError("❌ Invalid timestamp format in CSV")

df["time_sec"] = (
    df["timestamp"] - df["timestamp"].iloc[0]
).dt.total_seconds()

# =========================
# DROWSY FLAG (ROBUST)
# =========================
if "state" in df.columns:
    df["is_drowsy"] = df["state"].apply(
        lambda x: 1 if str(x).upper() == "DROWSY" else 0
    )
else:
    df["is_drowsy"] = df["ear"].apply(lambda x: 1 if x < 0.23 else 0)

# =========================
# YAWN FLAG
# =========================
if "yawning" in df.columns:
    df["is_yawn"] = df["yawning"].astype(int)
elif "yawns" in df.columns:
    df["is_yawn"] = df["yawns"].apply(lambda x: 1 if x > 0 else 0)
else:
    df["is_yawn"] = df["mar"].apply(lambda x: 1 if x > 0.40 else 0)

# =========================
# FATIGUE SCORE (SMOOTH)
# =========================
fatigue = []
score = 0.0

for i in range(len(df)):
    if df.loc[i, "ear"] < 0.23:
        score += 0.3

    if df.loc[i, "mar"] > 0.40:
        score += 0.4

    if df.loc[i, "is_drowsy"] == 1:
        score += 1.2

    score -= 0.15  # recovery
    score = max(0, min(score, 10))

    fatigue.append(score)

df["fatigue"] = fatigue

# =========================
# PLOT
# =========================
plt.figure(figsize=(12, 6))

plt.plot(df["time_sec"], df["fatigue"], label="Fatigue Score")
plt.plot(df["time_sec"], df["ear"] * 100, linestyle="--", label="EAR (scaled)")
plt.plot(df["time_sec"], df["mar"] * 100, linestyle="--", label="MAR (scaled)")

plt.xlabel("Time (seconds)")
plt.ylabel("Value")
plt.title("Driver Fatigue Timeline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
