import cv2
from face_detection import FaceDetector
from landmark_detection import FaceLandmarkDetector
from landmark_utils import LEFT_EYE, RIGHT_EYE, MOUTH
from feature_extraction import eye_aspect_ratio, mouth_aspect_ratio


EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7


def main():
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    landmark_detector = FaceLandmarkDetector()

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

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if ear < EAR_THRESHOLD:
                cv2.putText(frame, "DROWSY EYES!", (200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING!", (200, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow("Drowsiness Signals (Q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
