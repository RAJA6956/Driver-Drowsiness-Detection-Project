# src/main.py

import cv2
import time


def main():
    print("MAIN STARTED")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1.0)

    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    print("✅ Camera opened")

    cv2.namedWindow("Driver Drowsiness Detection", cv2.WINDOW_NORMAL)
    print("🟢 OpenCV window initialized")

    # Safe imports
    from face_detection import FaceDetector
    from landmark_detection import LandmarkDetector
    from feature_extraction import FeatureExtractor
    from alert_system import AlertSystem

    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    feature_extractor = FeatureExtractor()
    alert = AlertSystem()

    print("🟢 Modules initialized")
    print("🟢 Entering main loop")

    # =========================
    # EAR (EYES) LOGIC
    # =========================
    EAR_THRESHOLD = 0.23
    BLINK_FRAMES = 3
    DROWSY_FRAMES = 15

    ear_counter = 0
    blink_count = 0
    drowsy_state = "ALERT"

    # =========================
    # MAR (YAWN) LOGIC
    # =========================
    MAR_THRESHOLD = 0.40
    YAWN_FRAMES = 15

    mar_counter = 0
    yawn_count = 0
    yawning = False

    # =========================
    # FATIGUE SCORE (SAFE)
    # =========================
    fatigue_score = 0.0  # 0 → fresh, 100 → exhausted

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        face_rect = face_detector.detect(frame)

        if face_rect is not None:
            face_detector.draw(frame, face_rect)

            landmarks = landmark_detector.detect(frame, face_rect)

            if landmarks is not None:
                landmark_detector.draw(frame, landmarks)

                ear = feature_extractor.compute_ear(landmarks)
                mar = feature_extractor.compute_mar(landmarks)

                # =========================
                # EAR TEMPORAL LOGIC
                # =========================
                if ear < EAR_THRESHOLD:
                    ear_counter += 1
                else:
                    if ear_counter >= BLINK_FRAMES:
                        blink_count += 1
                        fatigue_score += 0.05  # small blink contribution
                    ear_counter = 0
                    drowsy_state = "ALERT"

                if ear_counter >= DROWSY_FRAMES:
                    drowsy_state = "DROWSY"
                    alert.play()
                    fatigue_score += 0.8  # strong fatigue signal

                # =========================
                # MAR TEMPORAL LOGIC (YAWN)
                # =========================
                if mar > MAR_THRESHOLD:
                    mar_counter += 1
                else:
                    if mar_counter >= YAWN_FRAMES:
                        yawn_count += 1
                        fatigue_score += 0.5  # moderate contribution
                    mar_counter = 0
                    yawning = False

                if mar_counter >= YAWN_FRAMES:
                    yawning = True

                # =========================
                # FATIGUE RECOVERY (VERY SLOW)
                # =========================
                if drowsy_state == "ALERT" and not yawning:
                    fatigue_score -= 0.1

                # Clamp score
                fatigue_score = max(0.0, min(100.0, fatigue_score))

                # =========================
                # VISUAL OVERLAY
                # =========================
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.putText(frame, f"Blinks: {blink_count}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.putText(frame, f"Yawns: {yawn_count}", (30, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                cv2.putText(
                    frame,
                    f"State: {drowsy_state}",
                    (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255) if drowsy_state == "DROWSY" else (0, 255, 0),
                    2,
                )

                # Fatigue color scale
                fatigue_color = (
                    (0, 255, 0) if fatigue_score < 30 else
                    (0, 255, 255) if fatigue_score < 60 else
                    (0, 0, 255)
                )

                cv2.putText(
                    frame,
                    f"Fatigue: {int(fatigue_score)}%",
                    (30, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    fatigue_color,
                    2,
                )

                if yawning:
                    cv2.putText(
                        frame,
                        "YAWNING",
                        (250, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3,
                    )

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🛑 Q pressed — exiting")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Clean shutdown")


if __name__ == "__main__":
    main()


