import cv2
import numpy as np
import json
import os
from detectors.utils import is_point_in_zone

class ForbiddenZoneDetector:
    def __init__(self, zones_json_path="config/no_bike_zones.json"):
        self.forbidden_zones = []
        self.bike_zone_violation_triggered_image_saved = False
        self.BIKE_CLASS_IDS = [1, 3]
        # 1: bicycle, 3: motorcycle

        if os.path.exists(zones_json_path):
            with open(zones_json_path, 'r') as f:
                raw_zones = json.load(f)
                for zone_coords in raw_zones:
                    self.forbidden_zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))
            print(f"Loaded {len(self.forbidden_zones)} forbidden zones from {zones_json_path}")
        else:
            print(f"Warning: No forbidden zones file found at '{zones_json_path}'. No bike zone detection will occur.")

    def process_frame(self, frame, detections, frame_count):
        """
        Detects if bikes are in forbidden zones and annotates the frame.
        Saves an image on the first detection within a forbidden zone.
        Args:
            frame (np.array): The current video frame.
            detections (list): List of dictionaries, each containing 'box' (x1,y1,x2,y2)
                                and 'cls' for all detected objects.
            frame_count (int): The current frame number.
        Returns:
            np.array: The annotated frame.
            bool: True if a bike zone violation occurred in this frame.
        """
        annotated_frame = frame.copy()
        bike_zone_violation_this_frame = False

        # Draw forbidden zones first
        for zone in self.forbidden_zones:
            cv2.polylines(annotated_frame, [zone], isClosed=True, color=(0, 255, 255), thickness=2) # Cyan color

        for d in detections:
            class_id = d['cls']
            if class_id in self.BIKE_CLASS_IDS:
                x1, y1, x2, y2 = d['box']
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                center = (cx, cy)
                
                label = d['label']
                conf = d['conf']

                bike_in_forbidden_zone = False
                for idx, zone in enumerate(self.forbidden_zones):
                    if is_point_in_zone(center, zone):
                        bike_in_forbidden_zone = True
                        break 

                if bike_in_forbidden_zone:
                    color = (0, 0, 255) 
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} ZONE VIOLATION! {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    bike_zone_violation_this_frame = True

                    if not self.bike_zone_violation_triggered_image_saved:
                        cv2.imwrite("triggered_frame.jpg", annotated_frame)
                        print(f"[ALERT] Bike zone violation detected at frame {frame_count} — image saved to 'triggered_frame.jpg'")
                        self.bike_zone_violation_triggered_image_saved = True
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated_frame, bike_zone_violation_this_frame