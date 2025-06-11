import cv2
import numpy as np
import math

# Constants for Human Pose Estimation
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
CONF_THRESHOLD = 0.3 

def is_point_in_zone(point, zone_polygon):
    """Checks if a point is inside a given polygon."""
    return cv2.pointPolygonTest(zone_polygon, point, False) >= 0

def calculate_keypoint_distance(p1, p2):
    """Calculates Euclidean distance between two keypoints."""
    if p1 is None or p2 is None:
        return float('inf')
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def visible(kp, threshold=CONF_THRESHOLD):
    """Checks if a keypoint is visible based on confidence."""
    return kp is not None and len(kp) > 2 and kp[2] > threshold

def get_midpoint(kp1, kp2):
    """Calculates the midpoint between two keypoints."""
    if visible(kp1) and visible(kp2):
        return [(kp1[0] + kp2[0]) / 2, (kp1[1] + kp2[1]) / 2, min(kp1[2], kp2[2])]
    return None

def get_vector(p1, p2):
    """Calculates a vector from point p1 to p2."""
    if p1 is None or p2 is None:
        return None
    return np.array([p2[0] - p1[0], p2[1] - p1[1]])

def get_angle(v1, v2):
    """Calculates the angle (degrees) between two vectors."""
    if v1 is None or v2 is None:
        return None
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    return np.degrees(np.arccos(np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)))

def get_person_height(kp):
    """Estimates person height based on keypoints."""
    if not all(visible(kp[i]) for i in [NOSE, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]):
        return None
    head_top = get_midpoint(kp[NOSE], get_midpoint(kp[LEFT_EAR], kp[RIGHT_EAR]))
    avg_ankle = get_midpoint(kp[LEFT_ANKLE], kp[RIGHT_ANKLE])
    if head_top and avg_ankle:
        return calculate_keypoint_distance(head_top, avg_ankle)
    return None

def calculate_centroid(kps):
    """Calculates the centroid of visible keypoints."""
    x_coords = [kp[0] for kp in kps if visible(kp)]
    y_coords = [kp[1] for kp in kps if visible(kp)]
    if not x_coords:
        return None
    return [sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords), 1.0]

def calculate_velocity(prev_kps, curr_kps, kp_idx, frame_width):
    """Calculates normalized velocity of a keypoint between frames."""
    if not prev_kps or not curr_kps or len(prev_kps) <= kp_idx or len(curr_kps) <= kp_idx:
        return 0.0
    if not visible(prev_kps[kp_idx]) or not visible(curr_kps[kp_idx]):
        return 0.0
    dist_pixels = calculate_keypoint_distance(prev_kps[kp_idx], curr_kps[kp_idx])
    return dist_pixels / frame_width