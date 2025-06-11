import cv2
from ultralytics import YOLO
import os
import json

from detectors.entry_zone_detector import EntryZoneDetector
from detectors.human_behavior_detector import HumanBehaviorDetector
from detectors.vandalism_detector import VandalismDetector
from detectors.forbidden_zone_detector import ForbiddenZoneDetector 

# Configuration
VIDEO_PATH = "/Users/bhavyakheni/Desktop/surveillance_project/videos/O2.mov" 
YOLO_N_MODEL_PATH = "/Users/bhavyakheni/Desktop/surveillance_project/models/yolov8n.pt"
YOLO_POSE_MODEL_PATH = "/Users/bhavyakheni/Desktop/surveillance_project/models/yolov8s-pose.pt"
NO_BIKE_ZONES_PATH = "/Users/bhavyakheni/Desktop/surveillance_project/config/no_bike_zones.json"
ENTRY_ZONES_PATH = "/Users/bhavyakheni/Desktop/surveillance_project/config/entry_zones.json" # Optional

# Load Models
try:
    general_object_model = YOLO(YOLO_N_MODEL_PATH)
    pose_estimation_model = YOLO(YOLO_POSE_MODEL_PATH)
    print("YOLO models loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO models: {e}. Make sure paths are correct and models exist.")
    exit()

# Initialize Detectors
entry_zones_data = None
if os.path.exists(ENTRY_ZONES_PATH):
    try:
        with open(ENTRY_ZONES_PATH, 'r') as f:
            entry_zones_data = json.load(f)
        print(f"Loaded entry zones from {ENTRY_ZONES_PATH}")
    except Exception as e:
        print(f"Error loading entry zones from {ENTRY_ZONES_PATH}: {e}")
        entry_zones_data = None 

entry_zone_detector = EntryZoneDetector(entry_zones_data=entry_zones_data)
human_behavior_detector = HumanBehaviorDetector()
vandalism_detector = VandalismDetector()
forbidden_zone_detector = ForbiddenZoneDetector(zones_json_path=NO_BIKE_ZONES_PATH)

# Video Capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_PATH}'. Please check the path.")
    exit()

frame_count = 0
cv2.namedWindow("Unified Surveillance System", cv2.WINDOW_NORMAL)

print("\nStarting video processing. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    frame_height, frame_width = frame.shape[:2]

    # 1. General Object Detection 
    results_general = general_object_model(frame, verbose=False)[0]
    
    all_detections_info = [] # Store all relevant detections for subsequent modules
    person_detections = []   # Specific for EntryZone and Vandalism
    car_detections = []      # Specific for Vandalism
    bike_detections_for_forbidden_zone = [] # For forbidden zone detector


    class_names = general_object_model.names

    for box in results_general.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = class_names.get(cls_id, f"unknown_class_{cls_id}")

        detection_data = {
            'box': (x1, y1, x2, y2),
            'cls': cls_id,
            'conf': conf,
            'label': label
        }
        all_detections_info.append(detection_data)

        if cls_id == 0: # Person
            person_detections.append(detection_data)
        elif cls_id == 2: # Car
            car_detections.append(detection_data)
        elif cls_id in [1, 3]: # Bicycle or Motorcycle
            bike_detections_for_forbidden_zone.append(detection_data)

    # 2. Person Pose Estimation
    person_detections_with_keypoints = []
    
    results_pose = pose_estimation_model(frame, verbose=False)[0]

    if results_pose.boxes is not None and results_pose.keypoints is not None:
        for i, box in enumerate(results_pose.boxes):

            if int(box.cls[0]) == 0: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                kps_xy = results_pose.keypoints.xy.cpu().numpy()[i]
                kps_conf = results_pose.keypoints.conf.cpu().numpy()[i]
                
                formatted_kps = []
                for j in range(len(kps_xy)):
                    formatted_kps.append([kps_xy[j][0], kps_xy[j][1], kps_conf[j]])

                person_detections_with_keypoints.append({
                    'box': (x1, y1, x2, y2),
                    'keypoints': formatted_kps,
                    'id': i
                })

    current_annotated_frame = frame.copy() 

    current_annotated_frame, bike_violation = forbidden_zone_detector.process_frame(
        current_annotated_frame, bike_detections_for_forbidden_zone, frame_count
    )

    current_annotated_frame = entry_zone_detector.process_frame(
        current_annotated_frame, person_detections
    )

    current_annotated_frame, stable_fight_alert = human_behavior_detector.process_frame(
        current_annotated_frame, person_detections_with_keypoints, frame_width, frame_height
    )
    if stable_fight_alert:
        cv2.putText(current_annotated_frame, "FIGHT ALERT!", (frame_width - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    current_annotated_frame = vandalism_detector.process_frame(
        current_annotated_frame, person_detections, car_detections
    )
    
    cv2.imshow("Unified Surveillance System", current_annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nUnified Surveillance System stopped.")