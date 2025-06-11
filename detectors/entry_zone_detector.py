import cv2
import numpy as np
from detectors.utils import is_point_in_zone

class EntryZoneDetector:
    def __init__(self, entry_zones_data=None):
        """
        Initializes the EntryZoneDetector with predefined entry zones.
        Args:
            entry_zones_data (list of list of [x, y]): List of polygons, where each polygon
                                                   is a list of [x, y] points.
        """
        self.entry_zones = []
        if entry_zones_data:
            for zone_coords in entry_zones_data:
                self.entry_zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))

        if not self.entry_zones:
            print("Warning: No entry zones provided. Using default example zones.")

    def process_frame(self, frame, person_detections):
        """
        Detects if people are entering predefined zones and annotates the frame.
        Args:
            frame (np.array): The current video frame.
            person_detections (list): List of dictionaries, each containing 'box' (x1,y1,x2,y2)
                                      and other info for detected persons.
        Returns:
            np.array: The annotated frame.
        """
        annotated_frame = frame.copy()
        
        for zone in self.entry_zones:
            cv2.polylines(annotated_frame, [zone], isClosed=True, color=(255, 255, 0), thickness=2)

        for person_data in person_detections:
            x1, y1, x2, y2 = person_data['box']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center = (cx, cy)

            for idx, zone in enumerate(self.entry_zones):
                if is_point_in_zone(center, zone):
                    print(f"  Person entered house zone {idx + 1}") 
                    cv2.putText(annotated_frame, f"Entered Zone {idx+1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return annotated_frame
    