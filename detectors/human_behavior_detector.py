import cv2
import numpy as np
from collections import deque
from detectors.utils import (
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    CONF_THRESHOLD, calculate_keypoint_distance, visible, get_midpoint,
    get_vector, get_angle, get_person_height, calculate_centroid, calculate_velocity
)

class HumanBehaviorDetector:
    def __init__(self):
        # Fight Detection Thresholds
        self.PROXIMITY_WRIST_HEAD_RATIO = 0.04
        self.PROXIMITY_LIMB_TORSO_RATIO = 0.06
        self.PROXIMITY_HIPS_RATIO = 0.10
        self.PROXIMITY_KNEES_ANKLES_RATIO = 0.08
        self.ARM_RAISED_ANGLE_THRESHOLD = 25
        self.ELBOW_PUNCH_BEND_ANGLE_MIN = 60
        self.ELBOW_PUNCH_BEND_ANGLE_MAX = 100
        self.KNEE_KICK_ANGLE_THRESHOLD = 90
        self.MIN_FIGHT_INDICATORS = 5
        self.FIGHT_DETECTION_WINDOW = 20
        self.VELOCITY_THRESHOLD_FIGHT = 0.04

        # Fall Detection Thresholds
        self.FALL_HEAD_TO_HIP_Y_RATIO = 0.55
        self.FALL_BODY_ALIGNMENT_X_DEVIATION = 0.25
        self.FALL_VERTICAL_DROP_RATIO = 0.08
        self.FALL_DETECTION_HISTORY_FRAMES = 15
        self.FALL_STILLNESS_THRESHOLD = 0.02
        self.MIN_FALL_INDICATORS = 2

        self.person_pose_histories = {} 
        self.fight_buffer = deque(maxlen=self.FIGHT_DETECTION_WINDOW)

    def _analyze_pose_for_fight(self, kp1, kp2, frame_width, frame_height, history1: deque, history2: deque):
        if kp1 is None or kp2 is None:
            return False

        indicators = 0

        def is_close(p1, p2, ratio_factor):
            if not visible(p1) or not visible(p2):
                return False
            return calculate_keypoint_distance(p1, p2) < frame_width * ratio_factor

        target_kps_head = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
        target_kps_upper_body = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        
        # 1. Direct Strikes/Attacks Proximity & Posture (Wrist/Elbow to Head/Torso)
        for shoulder_idx, elbow_idx, wrist_idx in [(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)]:
            if all(visible(kp1[idx]) for idx in [shoulder_idx, elbow_idx, wrist_idx]):
                arm_vec = get_vector(kp1[shoulder_idx], kp1[elbow_idx])
                forearm_vec = get_vector(kp1[elbow_idx], kp1[wrist_idx])
                if arm_vec is not None and forearm_vec is not None:
                    elbow_angle = get_angle(arm_vec, forearm_vec)
                    vertical_vec = np.array([0, -1])

                    if get_angle(arm_vec, vertical_vec) is not None and get_angle(arm_vec, vertical_vec) < self.ARM_RAISED_ANGLE_THRESHOLD and \
                       elbow_angle is not None and self.ELBOW_PUNCH_BEND_ANGLE_MIN < elbow_angle < self.ELBOW_PUNCH_BEND_ANGLE_MAX:
                        if kp1[wrist_idx][1] < kp1[shoulder_idx][1]:
                            for p2_target_idx in target_kps_head + target_kps_upper_body:
                                if is_close(kp1[wrist_idx], kp2[p2_target_idx], self.PROXIMITY_WRIST_HEAD_RATIO if p2_target_idx in target_kps_head else self.PROXIMITY_LIMB_TORSO_RATIO):
                                    indicators += 1.5

        for shoulder_idx, elbow_idx, wrist_idx in [(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)]:
            if all(visible(kp2[idx]) for idx in [shoulder_idx, elbow_idx, wrist_idx]):
                arm_vec = get_vector(kp2[shoulder_idx], kp2[elbow_idx])
                forearm_vec = get_vector(kp2[elbow_idx], kp2[wrist_idx])
                if arm_vec is not None and forearm_vec is not None:
                    elbow_angle = get_angle(arm_vec, forearm_vec)
                    vertical_vec = np.array([0, -1])

                    if get_angle(arm_vec, vertical_vec) is not None and get_angle(arm_vec, vertical_vec) < self.ARM_RAISED_ANGLE_THRESHOLD and \
                       elbow_angle is not None and self.ELBOW_PUNCH_BEND_ANGLE_MIN < elbow_angle < self.ELBOW_PUNCH_BEND_ANGLE_MAX:
                        if kp2[wrist_idx][1] < kp2[shoulder_idx][1]:
                            for p1_target_idx in target_kps_head + target_kps_upper_body:
                                if is_close(kp2[wrist_idx], kp1[p1_target_idx], self.PROXIMITY_WRIST_HEAD_RATIO if p1_target_idx in target_kps_head else self.PROXIMITY_LIMB_TORSO_RATIO):
                                    indicators += 1.5

        # 2. Kicking/Leg Attacks Proximity & Posture (Ankle/Knee to Body)
        target_kps_lower_body = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
        
        for hip_idx, knee_idx, ankle_idx in [(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]:
            if all(visible(kp1[idx]) for idx in [hip_idx, knee_idx, ankle_idx]):
                hip_knee_vec = get_vector(kp1[hip_idx], kp1[knee_idx])
                knee_ankle_vec = get_vector(kp1[knee_idx], kp1[ankle_idx])
                if hip_knee_vec is not None and knee_ankle_vec is not None:
                    leg_angle = get_angle(hip_knee_vec, knee_ankle_vec)
                    if leg_angle is not None and leg_angle < self.KNEE_KICK_ANGLE_THRESHOLD:
                        for p2_body_idx in target_kps_lower_body + target_kps_upper_body:
                            if is_close(kp1[ankle_idx], kp2[p2_body_idx], self.PROXIMITY_KNEES_ANKLES_RATIO):
                                indicators += 1.0

        for hip_idx, knee_idx, ankle_idx in [(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]:
            if all(visible(kp2[idx]) for idx in [hip_idx, knee_idx, ankle_idx]):
                hip_knee_vec = get_vector(kp2[hip_idx], kp2[knee_idx])
                knee_ankle_vec = get_vector(kp2[knee_idx], kp2[ankle_idx])
                if hip_knee_vec is not None and knee_ankle_vec is not None:
                    leg_angle = get_angle(hip_knee_vec, knee_ankle_vec)
                    if leg_angle is not None and leg_angle < self.KNEE_KICK_ANGLE_THRESHOLD:
                        for p1_body_idx in target_kps_lower_body + target_kps_upper_body:
                            if is_close(kp2[ankle_idx], kp1[p1_body_idx], self.PROXIMITY_KNEES_ANKLES_RATIO):
                                indicators += 1.0

        # 3. Close Body Proximity (Grappling/Pushing)
        mid1_hips = get_midpoint(kp1[LEFT_HIP], kp1[RIGHT_HIP])
        mid2_hips = get_midpoint(kp2[LEFT_HIP], kp2[RIGHT_HIP])
        if mid1_hips and mid2_hips and is_close(mid1_hips, mid2_hips, self.PROXIMITY_HIPS_RATIO):
            indicators += 0.5

        # 4. Body Orientation (Actively facing each other)
        mid1_shoulders = get_midpoint(kp1[LEFT_SHOULDER], kp1[RIGHT_SHOULDER])
        mid2_shoulders = get_midpoint(kp2[LEFT_SHOULDER], kp2[RIGHT_SHOULDER])
        
        if mid1_shoulders and mid2_shoulders:
            p1_center = calculate_centroid(kp1)
            p2_center = calculate_centroid(kp2)
            
            if p1_center and p2_center:
                vec_p1_shoulders_to_p2_center = get_vector(mid1_shoulders, p2_center)
                p1_forward_vec = get_vector(mid1_hips, mid1_shoulders) if mid1_hips else None

                vec_p2_shoulders_to_p1_center = get_vector(mid2_shoulders, p1_center)
                p2_forward_vec = get_vector(mid2_hips, mid2_shoulders) if mid2_hips else None

                if p1_forward_vec is not None and vec_p1_shoulders_to_p2_center is not None:
                    angle_p1_facing_p2 = get_angle(p1_forward_vec, vec_p1_shoulders_to_p2_center)
                    if angle_p1_facing_p2 is not None and angle_p1_facing_p2 < 30:
                        indicators += 0.5

                if p2_forward_vec is not None and vec_p2_shoulders_to_p1_center is not None:
                    angle_p2_facing_p1 = get_angle(p2_forward_vec, vec_p2_shoulders_to_p1_center)
                    if angle_p2_facing_p1 is not None and angle_p2_facing_p1 < 30:
                        indicators += 0.5

        # 5. Aggressive Limb Velocity with Direction (Requires history)
        if len(history1) >= 2 and len(history2) >= 2:
            prev_kp1 = history1[-2]
            prev_kp2 = history2[-2]

            for p1_limb_idx in [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]:
                if visible(kp1[p1_limb_idx]) and visible(prev_kp1[p1_limb_idx]):
                    velocity_p1_limb = calculate_velocity(prev_kp1, kp1, p1_limb_idx, frame_width)
                    if velocity_p1_limb > self.VELOCITY_THRESHOLD_FIGHT:
                        limb_movement_vec = get_vector(prev_kp1[p1_limb_idx], kp1[p1_limb_idx])
                        
                        p2_target_center = get_midpoint(get_midpoint(kp2[LEFT_SHOULDER], kp2[RIGHT_SHOULDER]), kp2[NOSE])
                        if p2_target_center is None: p2_target_center = kp2[NOSE]
                        
                        if p2_target_center and limb_movement_vec is not None:
                            limb_to_target_vec = get_vector(kp1[p1_limb_idx], p2_target_center)
                            if limb_to_target_vec is not None:
                                angle_to_target = get_angle(limb_movement_vec, limb_to_target_vec)
                                if angle_to_target is not None and angle_to_target < 20:
                                    indicators += 1.0

            for p2_limb_idx in [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]:
                if visible(kp2[p2_limb_idx]) and visible(prev_kp2[p2_limb_idx]):
                    velocity_p2_limb = calculate_velocity(prev_kp2, kp2, p2_limb_idx, frame_width)
                    if velocity_p2_limb > self.VELOCITY_THRESHOLD_FIGHT:
                        limb_movement_vec = get_vector(prev_kp2[p2_limb_idx], kp2[p2_limb_idx])
                        
                        p1_target_center = get_midpoint(get_midpoint(kp1[LEFT_SHOULDER], kp1[RIGHT_SHOULDER]), kp1[NOSE])
                        if p1_target_center is None: p1_target_center = kp1[NOSE]
                        
                        if p1_target_center and limb_movement_vec is not None:
                            limb_to_target_vec = get_vector(kp2[p2_limb_idx], p1_target_center)
                            if limb_to_target_vec is not None:
                                angle_to_target = get_angle(limb_movement_vec, limb_to_target_vec)
                                if angle_to_target is not None and angle_to_target < 20:
                                    indicators += 1.0

        return indicators >= self.MIN_FIGHT_INDICATORS

    def _detect_fall(self, current_kp, frame_width, frame_height, pose_history: deque):
        if current_kp is None:
            return False

        pose_history.append(current_kp)

        if len(pose_history) < 2:
            return False

        fall_indicators = 0

        current_head = get_midpoint(current_kp[NOSE], get_midpoint(current_kp[LEFT_EAR], current_kp[RIGHT_EAR]))
        current_neck = get_midpoint(current_kp[LEFT_SHOULDER], current_kp[RIGHT_SHOULDER])
        current_hips = get_midpoint(current_kp[LEFT_HIP], current_kp[RIGHT_HIP])
        current_torso_center = get_midpoint(current_neck, current_hips)

        if current_head and current_hips and visible(current_head) and visible(current_hips) and current_head[1] > current_hips[1]:
            fall_indicators += 1

        if current_neck and current_hips and visible(current_neck) and visible(current_hips):
            body_vec = get_vector(current_neck, current_hips)
            if body_vec is not None:
                vertical_orientation = np.array([0, 1])
                angle = get_angle(body_vec, vertical_orientation)
                if angle is not None and angle > 90 - (90 * self.FALL_BODY_ALIGNMENT_X_DEVIATION):
                    fall_indicators += 1

        if len(pose_history) >= self.FALL_DETECTION_HISTORY_FRAMES:
            oldest_kp = pose_history[0]
            oldest_torso_center = get_midpoint(get_midpoint(oldest_kp[LEFT_SHOULDER], oldest_kp[RIGHT_SHOULDER]),
                                               get_midpoint(oldest_kp[LEFT_HIP], oldest_kp[RIGHT_HIP]))

            if current_torso_center and oldest_torso_center and visible(current_torso_center) and visible(oldest_torso_center):
                vertical_drop = current_torso_center[1] - oldest_torso_center[1]
                person_height = get_person_height(current_kp)

                if person_height and vertical_drop > person_height * self.FALL_VERTICAL_DROP_RATIO:
                    fall_indicators += 1

        if fall_indicators >= 1 and len(pose_history) >= self.FALL_DETECTION_HISTORY_FRAMES:
            is_still = True
            for i in range(1, self.FALL_DETECTION_HISTORY_FRAMES):
                prev_kp_hist = pose_history[i-1]
                curr_kp_hist = pose_history[i]
                
                prev_centroid = calculate_centroid(prev_kp_hist)
                curr_centroid = calculate_centroid(curr_kp_hist)

                if prev_centroid is None or curr_centroid is None:
                    is_still = False
                    break
                
                displacement = calculate_keypoint_distance(prev_centroid, curr_centroid) / frame_width
                if displacement > self.FALL_STILLNESS_THRESHOLD:
                    is_still = False
                    break
            
            if is_still:
                fall_indicators += 1

        return fall_indicators >= self.MIN_FALL_INDICATORS

    def process_frame(self, frame, person_detections_with_keypoints, frame_width, frame_height):
        """
        Detects fight and fall behaviors and annotates the frame.
        Args:
            frame (np.array): The current video frame.
            person_detections_with_keypoints (list): List of dictionaries, each containing
                                                      'box' (x1,y1,x2,y2), 'keypoints', and 'id'
                                                      for detected persons.
            frame_width (int): Width of the frame.
            frame_height (int): Height of the frame.
        Returns:
            np.array: The annotated frame.
            bool: True if stable fight detected, False otherwise.
        """
        annotated_frame = frame.copy()
        
        fight_detected_this_frame = False
        
        # Process fight detection
        for i in range(len(person_detections_with_keypoints)):
            for j in range(i + 1, len(person_detections_with_keypoints)):
                p1_data, p2_data = person_detections_with_keypoints[i], person_detections_with_keypoints[j]
                
                # Check for proximity before detailed fight analysis (optimization)
                center1 = ((p1_data['box'][0] + p1_data['box'][2]) // 2, (p1_data['box'][1] + p1_data['box'][3]) // 2)
                center2 = ((p2_data['box'][0] + p2_data['box'][2]) // 2, (p2_data['box'][1] + p2_data['box'][3]) // 2)
                dist_centers = calculate_keypoint_distance(center1, center2)

                if dist_centers < frame_width * 0.4: # Only analyze if relatively close
                    # Ensure histories exist for these persons
                    if p1_data['id'] not in self.person_pose_histories:
                        self.person_pose_histories[p1_data['id']] = deque(maxlen=self.FALL_DETECTION_HISTORY_FRAMES)
                    if p2_data['id'] not in self.person_pose_histories:
                        self.person_pose_histories[p2_data['id']] = deque(maxlen=self.FALL_DETECTION_HISTORY_FRAMES)

                    if self._analyze_pose_for_fight(p1_data['keypoints'], p2_data['keypoints'], frame_width, frame_height,
                                                   self.person_pose_histories[p1_data['id']], self.person_pose_histories[p2_data['id']]):
                        fight_detected_this_frame = True
                        # Draw boxes and text for fighting individuals
                        cv2.rectangle(annotated_frame, (p1_data['box'][0], p1_data['box'][1]), (p1_data['box'][2], p1_data['box'][3]), (0, 0, 255), 2)
                        cv2.rectangle(annotated_frame, (p2_data['box'][0], p2_data['box'][1]), (p2_data['box'][2], p2_data['box'][3]), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "FIGHT",
                                    (min(p1_data['box'][0], p2_data['box'][0]), min(p1_data['box'][1], p2_data['box'][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Update fight buffer for stable fight alert
        self.fight_buffer.append(fight_detected_this_frame)
        if sum(self.fight_buffer) > self.FIGHT_DETECTION_WINDOW * 0.6:
            cv2.putText(annotated_frame, "Stable Fight Alert!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


        # Process fall detection for each person
        for person_data in person_detections_with_keypoints:
            person_id = person_data['id']
            kp = person_data['keypoints']

            if person_id not in self.person_pose_histories:
                self.person_pose_histories[person_id] = deque(maxlen=self.FALL_DETECTION_HISTORY_FRAMES)
            
            fall_detected = self._detect_fall(kp, frame_width, frame_height, self.person_pose_histories[person_id])
            
            if fall_detected:
                x1, y1, x2, y2 = person_data['box']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "FALL", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw keypoints for all detected persons (optional, can be removed for cleaner output)
        for person_data in person_detections_with_keypoints:
            kps = person_data['keypoints']
            if kps:
                for kp in kps:
                    if visible(kp):
                        cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

        return annotated_frame, sum(self.fight_buffer) > self.FIGHT_DETECTION_WINDOW * 0.6 # Return stable fight alert