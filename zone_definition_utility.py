import cv2
import numpy as np
import json
import os

def define_zones_from_video_frame(video_path, output_image_path="/Users/bhavyakheni/Desktop/surveillance_project/video_frame_for_zones.jpg", output_json_path="/Users/bhavyakheni/Desktop/surveillance_project/config/no_bike_zones.json"):
    """
    Captures the first frame from a video, allows user to draw polygons on it,
    and saves the drawn polygon points to a JSON file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'. Please check the path.")
        return

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from video '{video_path}'. Is the video empty?")
        cap.release()
        return

    cap.release()

    if not os.path.exists('config'):
        os.makedirs('config')

    # Save the captured frame for user interaction
    cv2.imwrite(output_image_path, frame)
    print(f"Captured frame saved to '{output_image_path}'.")
    print(f"Please use this image to define your no-bike zones.")

    img = cv2.imread(output_image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {output_image_path}. Check the path!")

    clone = img.copy()
    current_points = []
    all_zones = []

    def click_event(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

            if len(current_points) > 1:
                cv2.line(img, current_points[-2], current_points[-1], (0, 0, 255), 2)
            cv2.imshow("Define No-Bike Zones", img)

    cv2.imshow("Define No-Bike Zones", img)
    cv2.setMouseCallback("Define No-Bike Zones", click_event)

    print("\nInstructions for Defining Zones:")
    print("  Click to define polygon points.")
    print("  Press 'c' to close the current polygon and add it to zones.")
    print("  Press 'n' to start defining a NEW polygon.")
    print("  Press 'r' to reset ALL zones and start over.")
    print("  Press 's' to SAVE all defined zones and quit.")
    print("  Press 'q' to quit WITHOUT saving.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and len(current_points) >= 3:
            
            cv2.line(img, current_points[-1], current_points[0], (0, 0, 255), 2)
            pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.imshow("Define No-Bike Zones", img)
            
            all_zones.append(current_points)
            current_points = [] 
            print(f"  Polygon {len(all_zones)} defined. Press 'n' for new, 's' to save, 'q' to quit.")

        elif key == ord('n'):
            if len(current_points) > 0: 
                print("  Current polygon not closed. Press 'c' to close it first or 'r' to reset.")
            else:
                print("  Starting a new polygon...")

        elif key == ord('r'):
            img = clone.copy()
            current_points.clear()
            all_zones.clear()
            cv2.imshow("Define No-Bike Zones", img)
            print("  All zones reset. Start drawing new polygons.")

        elif key == ord('s'):
            if len(current_points) >= 3: 
                cv2.line(img, current_points[-1], current_points[0], (0, 0, 255), 2)
                pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                all_zones.append(current_points)
                cv2.imshow("Define No-Bike Zones", img)
            elif len(current_points) > 0:
                 print("  Cannot save unclosed polygon with less than 3 points.")

            if all_zones:
                with open(output_json_path, 'w') as f:
                    json.dump(all_zones, f)
                print(f"  Zones saved to '{output_json_path}'.")
            else:
                print("  No zones defined to save.")
            break

        elif key == ord('q'):
            print("  Exiting without saving zones.")
            break

    cv2.destroyAllWindows()
    print("\nZone definition process complete.")

if __name__ == "__main__":
    video_to_use_for_zones = "/Users/bhavyakheni/Desktop/surveillance_project/videos/B.mp4" 
    define_zones_from_video_frame(video_to_use_for_zones)