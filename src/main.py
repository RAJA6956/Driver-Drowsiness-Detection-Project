import cv2
print("MAIN FILE LOADED")

def safe_import(name, statement):
    try:
        exec(statement, globals())
        print(f"✅ Imported {name}")
    except Exception as e:
        print(f"❌ FAILED importing {name}")
        print(e)
        exit(1)

# 🔍 Import files ONE BY ONE
safe_import("face_detection", "from face_detection import FaceDetector")
safe_import("landmark_detection", "from landmark_detection import LandmarkDetector")
safe_import("feature_extraction", "from feature_extraction import FeatureExtractor")
safe_import("temporal_logic", "from temporal_logic import TemporalLogic")
safe_import("alert_system", "from alert_system import AlertSystem")
safe_import("logger", "from logger import SessionLogger")


def main():
    print("INSIDE MAIN")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("Camera failed")
        return

    print("Starting camera loop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("CAMERA TEST", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("__main__ confirmed")
    main()
