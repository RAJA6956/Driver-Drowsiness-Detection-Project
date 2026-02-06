# src/main.py

import cv2
import time
import traceback


def main():
    print("MAIN STARTED")

    # =========================
    # CAMERA SETUP
    # =========================
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(1.5)

    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    print("✅ Camera opened")

    # Force OpenCV GUI initialization
    ret, frame = cap.read()
    if ret and frame is not None:
        cv2.imshow("Driver Drowsiness Detection", frame)
        cv2.waitKey(1)

    print("🟢 OpenCV window initialized")

    # =========================
    # IMPORT MODULES
    # =========================
    try:
        from face_detection import FaceDetector
        from landmark_detection import LandmarkDetector
        print("✅ FaceDetector & LandmarkDetector imported")
    except Exception:
        print("❌ Import failed")
        traceback.print_exc()
        return

    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()

    print("🟢 Detectors initialized")
    print("🟢 Entering main loop")

    # =========================
    # MAIN LOOP
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Status text (always visible)
        cv2.putText(
            frame,
            "FACE + LANDMARK STAGE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        try:
            # Face detection
            face_rect = face_detector.detect(frame)
            if face_rect is not None:
                face_detector.draw(frame, face_rect)

                # Landmark detection
                landmarks = landmark_detector.detect(frame, face_rect)
                if landmarks is not None:
                    landmark_detector.draw(frame, landmarks)

        except Exception:
            print("❌ Crash inside detection pipeline")
            traceback.print_exc()

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # =========================
    # CLEAN EXIT
    # =========================
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Exit clean")


if __name__ == "__main__":
    main()
