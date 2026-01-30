import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

# --- 1. SETUP & MODEL DOWNLOAD ---
# The new API requires a model file. We download it if you don't have it.
MODEL_PATH = 'pose_landmarker_lite.task'
if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_PATH}...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

# Configuration
VIDEO_PATH = 'motion-1.mp4'  # Replace with your video
OUTPUT_PATH = 'output_video.mp4'

# Import new Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def main():
    # Create the PoseLandmarker instance
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1  # Detect 1 person
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Cannot open {VIDEO_PATH}")
            return

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

        print("Processing video...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 1. MediaPipe Tasks requires an MPImage
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # 2. We must provide the timestamp in milliseconds
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # 3. Detect
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # 4. Draw & Label
            # If poses are detected
            if detection_result.pose_landmarks:
                # Get the first person's landmarks
                landmarks = detection_result.pose_landmarks[0]
                
                # Helper to get pixel coordinates
                def get_px(lm):
                    return int(lm.x * width), int(lm.y * height)

                # --- Draw Skeleton Connections (Manual, since drawing_utils is part of solutions) ---
                # Defining simple pairs for drawing lines
                connections = [
                    (11, 12), (11, 13), (13, 15), # Left Arm
                    (12, 14), (14, 16),           # Right Arm
                    (11, 23), (12, 24), (23, 24), # Torso
                    (23, 25), (25, 27),           # Left Leg
                    (24, 26), (26, 28)            # Right Leg
                ]
                
                # Draw Lines
                for idx1, idx2 in connections:
                    pt1 = get_px(landmarks[idx1])
                    pt2 = get_px(landmarks[idx2])
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

                # --- Label Specific Joints ---
                # Map indices to names (Standard MediaPipe Indices)
                # 0:nose, 11:left_shoulder, 12:right_shoulder, 13:left_elbow, 14:right_elbow
                # 15:left_wrist, 16:right_wrist, 23:left_hip, 24:right_hip, 25:left_knee, 26:right_knee
                # 27:left_ankle, 28:right_ankle
                
                mapping = {
                    "Head": 0,
                    "L.Shldr": 11, "R.Shldr": 12,
                    "L.Elbow": 13, "R.Elbow": 14,
                    "L.Hand": 15,  "R.Hand": 16,
                    "L.Waist": 23, "R.Waist": 24,
                    "L.Knee": 25,  "R.Knee": 26,
                    "L.Foot": 27,  "R.Foot": 28
                }

                # Label the standard points
                for label, idx in mapping.items():
                    px, py = get_px(landmarks[idx])
                    # Draw dot
                    cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                    # Draw text
                    cv2.putText(frame, label, (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # --- Label Chest (Midpoint) ---
                l_sh = landmarks[11]
                r_sh = landmarks[12]
                chest_x = int((l_sh.x + r_sh.x) / 2 * width)
                chest_y = int((l_sh.y + r_sh.y) / 2 * height)
                
                cv2.circle(frame, (chest_x, chest_y), 6, (0, 255, 255), -1)
                cv2.putText(frame, "Chest", (chest_x+10, chest_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # --- Label Hand Coordinates ---
                l_hand = get_px(landmarks[15])
                r_hand = get_px(landmarks[16])
                
                cv2.putText(frame, f"L:{l_hand}", (l_hand[0]-40, l_hand[1]+30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"R:{r_hand}", (r_hand[0]-40, r_hand[1]+30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            
            # Show preview
            cv2.imshow("New MediaPipe Tasks", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()