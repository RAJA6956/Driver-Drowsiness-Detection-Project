

**Driver Drowsiness Detection System**



A real-time computer vision system that monitors a driver’s alertness using facial landmarks and physiological indicators such as eye aspect ratio (EAR) and mouth aspect ratio (MAR).

The system detects blinks, yawns, and drowsy behavior, logs session data, and enables post-session fatigue analysis.



** Features**



Real-time face detection using webcam



Eye and mouth landmark detection



EAR-based blink \& drowsiness detection



MAR-based yawning detection



Temporal logic (frame-based noise reduction)



Audio alert for drowsiness



CSV session logging



Post-session fatigue timeline visualization



** System Overview**



The system follows a rule-based, explainable pipeline optimized for real-time use:



Capture video stream from webcam



Detect face region



Extract facial landmarks (eyes \& mouth)



Compute EAR and MAR values



Apply temporal logic to detect:



Blinks



Yawns



Drowsy state



Trigger alert if drowsiness persists



Log session data to CSV for offline analysis



This design avoids heavy ML models and is suitable for edge or embedded systems.



** Metrics Used**

Eye Aspect Ratio (EAR)



Indicates eye openness



Low EAR over consecutive frames → drowsiness



Mouth Aspect Ratio (MAR)



Indicates mouth openness



High MAR over time → yawning



Fatigue Score (Offline)



Derived from EAR, MAR, and drowsy events



Used for post-session trend analysis (not real-time classification)



**🗂 Project Structure**

Driver-Drowsiness-Detection-Project/

│

├── src/

│   ├── main.py                 # Real-time detection pipeline

│   ├── face\_detection.py       # Face detection module

│   ├── landmark\_detection.py   # Facial landmarks

│   ├── feature\_extraction.py   # EAR / MAR computation

│   ├── alert\_system.py         # Audio alert

│   └── plot\_timeline.py        # CSV → fatigue graph

│

├── logs/

│   └── session\_log.csv         # Session data

│

├── requirements.txt

└── README.md



** Installation**



Create and activate a virtual environment (Python 3.8 recommended):



python -m venv venv38

venv38\\Scripts\\activate





Install dependencies:



pip install -r requirements.txt



**Usage**

Run real-time detection

python src/main.py





Press Q to exit safely



Session data is saved to logs/session\_log.csv



Plot fatigue timeline

python src/plot\_timeline.py





Displays:



Fatigue score over time



EAR and MAR trends



** Data Logging**



Each session logs:



Timestamp



EAR value



MAR value



Blink count



Yawn count



Driver state



The CSV enables offline analysis and visualization without affecting real-time performance.



** Why Rule-Based (Not ML)?**



Fully explainable decisions



Lightweight and fast



No training data required



Robust for real-time deployment



Suitable for embedded systems



**🔮 Future Improvements**



Risk level classification from fatigue score



Driver identity calibration



Night-time robustness



Embedded deployment (Raspberry Pi)



** Author**



**Raja Kaushal**

**B.Tech CSE (Data Science)**

**Galgotias University**

