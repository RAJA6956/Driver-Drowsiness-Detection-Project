import cv2
from landmark_detection import FaceLandmarkDetector
from landmark_utils import LEFT_EYE, RIGHT_EYE, MOUTH
from feature_extraction import eye_aspect_ratio, mouth_aspect_ratio
from temporal_logic import DrowsinessMonitor
from alert_system import AlertSystem


MAR_THRESHOLD = 0.7


def main():
    cap = cv2.VideoCapture(0)
    landmark_detector = FaceLandmarkDetector()
    monitor = DrowsinessMonitor()
    alert = AlertSystem()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = landmark_detector.detect_landmarks(frame)

        if landmarks:
            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]
            mouth = [landmarks[i] for i in MOUTH]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            mar = mouth_aspect_ratio(mouth)

            state = monitor.update(ear)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Blinks: {monitor.blink_count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if state == "DROWSY":
                cv2.putText(frame, "DROWSINESS ALERT!", (120, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                alert.trigger_alert()

            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING", (120, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Detection with Alerts (Q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


