import cv2
from face_detection import FaceDetector
from landmark_detection import FaceLandmarkDetector
from landmark_utils import LEFT_EYE, RIGHT_EYE, MOUTH


def draw_points(frame, points):
    for (x, y) in points:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def main():
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    landmark_detector = FaceLandmarkDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect_faces(frame)
        landmarks = landmark_detector.detect_landmarks(frame)

        if landmarks:
            draw_points(frame, [landmarks[i] for i in LEFT_EYE])
            draw_points(frame, [landmarks[i] for i in RIGHT_EYE])
            draw_points(frame, [landmarks[i] for i in MOUTH])

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Landmarks - Eyes & Mouth (Q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
