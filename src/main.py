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
    # TEMPORAL LOGIC VARIABLES
    # =========================
    EAR_THRESHOLD = 0.23
    BLINK_FRAMES = 3
    DROWSY_FRAMES = 15

    ear_counter = 0
    blink_count = 0
    state = "ALERT"

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
                # TEMPORAL LOGIC
                # =========================
                if ear < EAR_THRESHOLD:
                    ear_counter += 1
                else:
                    if ear_counter >= BLINK_FRAMES:
                        blink_count += 1
                    ear_counter = 0
                    state = "ALERT"

                if ear_counter >= DROWSY_FRAMES:
                    state = "DROWSY"
                    alert.play()   # 🔊 SAFE AUDIO TRIGGER

                # =========================
                # VISUAL DEBUG
                # =========================
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.putText(frame, f"Blinks: {blink_count}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.putText(frame, f"State: {state}", (30, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if state == "DROWSY" else (0, 255, 0), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("🛑 Q pressed — exiting")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Clean shutdown")


if __name__ == "__main__":
    main()
