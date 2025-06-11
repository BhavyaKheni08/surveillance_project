import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class VandalismDetector:
    def __init__(self, dist_thresh=100, motion_thresh=1500, max_age=30, conf_thresh=0.3):
        """
        Initializes the VandalismDetector.
        Args:
            dist_thresh (int): Maximum distance (pixels) for a person to be considered "near" a car.
            motion_thresh (int): Minimum motion area (pixels) within a person's bounding box to trigger vandalism alert.
            max_age (int): Maximum number of frames a track can be without detections before it's deleted.
            conf_thresh (float): Confidence threshold for YOLO detections.
        """
        self.tracker = DeepSort(max_age=max_age)
        self.prev_gray = None
        self.DIST_THRESH = dist_thresh
        self.MOTION_THRESH = motion_thresh
        self.CONF_THRESH = conf_thresh

        self.PERSON_CLASS_ID = 0
        self.CAR_CLASS_ID = 2

    def preprocess_frame(self, frame):
        """Converts frame to grayscale and applies Gaussian blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        return blur

    def calculate_motion(self, current_blur, bbox):
        """Calculates motion within a given bounding box."""
        if self.prev_gray is None:
            return 0

        diff = cv2.absdiff(self.prev_gray, current_blur)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x1, y1, x2, y2 = map(int, bbox)
        motion_area = 0
        for c in contours:
            if cv2.contourArea(c) < 50:
                continue
            cx, cy, cw, ch = cv2.boundingRect(c)

            if x1 < cx + cw // 2 < x2 and y1 < cy + ch // 2 < y2:
                motion_area += cv2.contourArea(c)
        return motion_area

    def process_frame(self, frame, person_detections, car_detections):
        """
        Detects potential vandalism based on person proximity to cars and motion.
        Args:
            frame (np.array): The current video frame.
            person_detections (list): List of dictionaries, each containing 'box' (x1,y1,x2,y2)
                                      and other info for detected persons.
            car_detections (list): List of dictionaries, each containing 'box' (x1,y1,x2,y2)
                                   for detected cars.
        Returns:
            np.array: The annotated frame.
        """
        annotated_frame = frame.copy()
        current_blur = self.preprocess_frame(annotated_frame)

        if self.prev_gray is None:
            self.prev_gray = current_blur
            return annotated_frame 
        
        ds_person_detections = []
        for p_data in person_detections:
            x1, y1, x2, y2 = p_data['box']
            conf = p_data['conf'] 
            ds_person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = self.tracker.update_tracks(ds_person_detections, frame=annotated_frame)

        car_centers = [((c_data['box'][0] + c_data['box'][2]) // 2, (c_data['box'][1] + c_data['box'][3]) // 2)
                       for c_data in car_detections]

        for t in tracks:
            if not t.is_confirmed():
                continue

            x1, y1, x2, y2 = t.to_ltrb()
            track_id = t.track_id
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            near_car = any(np.hypot(cx - ccx, cy - ccy) < self.DIST_THRESH for ccx, ccy in car_centers)

            motion = self.calculate_motion(current_blur, (x1, y1, x2, y2))

            color, label = (0, 255, 0), f"ID {track_id}"
            vandalism_detected = False
            if near_car and motion > self.MOTION_THRESH:
                color, label = (0, 0, 255), f"VANDALISM! ID {track_id}"
                vandalism_detected = True

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if vandalism_detected:
                pass 

        self.prev_gray = current_blur
        return annotated_frame