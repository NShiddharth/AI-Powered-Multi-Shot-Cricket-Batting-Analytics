"""
Multi-Shot Reference Video Generator
Generates realistic, physically-sound reference video clips for:
- Straight Drive (straight_0001.avi to straight_0010.avi)
- Pull Shot (pull_0001.avi to pull_0010.avi)
Features full batter body keypoints, bat motion, ball trajectory, and realistic kinematics.
"""

import os
import cv2
import numpy as np

def draw_cricket_scene(frame, pitch_color=(95, 145, 95)):
    # Field background
    frame[:] = pitch_color
    # Pitch strip in center
    cv2.rectangle(frame, (350, 200), (930, 719), (130, 160, 185), -1)
    # Popping crease
    cv2.line(frame, (300, 560), (980, 560), (240, 240, 240), 3)
    # Bowling crease
    cv2.line(frame, (340, 580), (940, 580), (240, 240, 240), 2)
    # Stumps
    for sx in [620, 640, 660]:
        cv2.line(frame, (sx, 580), (sx, 480), (220, 180, 100), 5)
    cv2.line(frame, (615, 480), (665, 480), (220, 180, 100), 4)

def render_batter_frame(w, h, shot_type, t_norm, var_seed=0):
    """
    Renders an articulated batter with realistic cricket proportions.
    t_norm ranges from 0.0 (stance) to 1.0 (follow-through completion).
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    draw_cricket_scene(frame)
    rng = np.random.RandomState(var_seed)
    noise_x = rng.uniform(-4, 4)
    noise_y = rng.uniform(-3, 3)

    cx = 580 + noise_x
    cy = 400 + noise_y

    # Interpolate shot kinematics
    if shot_type == "straight_drive":
        # Straight drive kinematics:
        # High leading elbow, straight bat swing down the pitch, front foot forward
        if t_norm < 0.25: # Stance & backlift
            progress = t_norm / 0.25
            stride_x = 0
            stride_y = 0
            bat_angle = 30 + 35 * progress
            elbow_deg = 80 - 10 * progress
            head_x = cx
            head_y = cy - 130
        elif t_norm < 0.65: # Downswing into impact
            progress = (t_norm - 0.25) / 0.40
            stride_x = 75 * progress
            stride_y = 35 * progress
            bat_angle = 65 - 110 * progress
            elbow_deg = 70 + 75 * progress # Reaches ~145 deg at contact
            head_x = cx + stride_x * 0.45
            head_y = cy - 120 + 20 * progress
        else: # High follow-through straight
            progress = (t_norm - 0.65) / 0.35
            stride_x = 75
            stride_y = 35
            bat_angle = -45 - 65 * progress
            elbow_deg = 145 + 10 * progress
            head_x = cx + stride_x * 0.45
            head_y = cy - 100

        # Joint coordinates for straight drive
        nose = (int(head_x), int(head_y))
        l_sh = (int(head_x - 30), int(head_y + 40))
        r_sh = (int(head_x + 30), int(head_y + 38))
        l_hip = (int(head_x - 20), int(head_y + 160))
        r_hip = (int(head_x + 25), int(head_y + 155))

        # Stride to front knee/ankle
        l_knee = (int(l_hip[0] + stride_x * 0.6), int(l_hip[1] + 90 + stride_y * 0.4))
        l_ankle = (int(l_hip[0] + stride_x), int(l_knee[1] + 85))
        l_foot = (int(l_ankle[0] + 30), int(l_ankle[1] + 8)) # Straight forward direction (~95 deg)

        # Back leg
        r_knee = (int(r_hip[0] - 15), int(r_hip[1] + 95))
        r_ankle = (int(r_knee[0] - 10), int(r_knee[1] + 85))
        r_foot = (int(r_ankle[0] + 20), int(r_ankle[1] + 5))

        # Arms & Bat
        rad = np.radians(elbow_deg)
        l_elbow = (int(l_sh[0] + 55 * np.cos(rad)), int(l_sh[1] + 55 * np.sin(rad)))
        l_wrist = (int(l_elbow[0] + 50 * np.cos(rad - 0.3)), int(l_elbow[1] + 50 * np.sin(rad - 0.3)))
        r_elbow = (int(r_sh[0] + 45), int(r_sh[1] + 50))
        r_wrist = (int(l_wrist[0] + 15), int(l_wrist[1] - 10))

        # Bat blade
        bat_rad = np.radians(bat_angle)
        bat_end = (int(l_wrist[0] + 90 * np.cos(bat_rad)), int(l_wrist[1] + 90 * np.sin(bat_rad)))

    elif shot_type == "pull_shot":
        # Pull shot kinematics:
        # Cross-bat swing, horizontal plane, back-foot pivot, high trunk rotation
        if t_norm < 0.25: # Stance & rock back
            progress = t_norm / 0.25
            back_shift = -30 * progress
            bat_angle = 45 + 40 * progress
            rot_deg = 20 * progress
            head_x = cx + back_shift
            head_y = cy - 130
        elif t_norm < 0.65: # Horizontal cross-bat pull stroke
            progress = (t_norm - 0.25) / 0.40
            back_shift = -30 - 10 * progress
            bat_angle = 85 - 130 * progress
            rot_deg = 20 + 80 * progress # Strong trunk rotation
            head_x = cx + back_shift
            head_y = cy - 130 + 10 * progress
        else: # Follow-through wrap to leg-side
            progress = (t_norm - 0.65) / 0.35
            back_shift = -40
            bat_angle = -45 - 80 * progress
            rot_deg = 100 + 40 * progress
            head_x = cx + back_shift
            head_y = cy - 120

        rot_rad = np.radians(rot_deg)
        sh_offset_x = 35 * np.cos(rot_rad)
        sh_offset_y = 15 * np.sin(rot_rad)

        nose = (int(head_x), int(head_y))
        l_sh = (int(head_x - sh_offset_x), int(head_y + 40 - sh_offset_y))
        r_sh = (int(head_x + sh_offset_x), int(head_y + 40 + sh_offset_y))
        l_hip = (int(head_x - 20), int(head_y + 160))
        r_hip = (int(head_x + 25), int(head_y + 155))

        # Weight on back foot, front leg opened
        r_knee = (int(r_hip[0] - 5), int(r_hip[1] + 90))
        r_ankle = (int(r_knee[0] - 5), int(r_knee[1] + 85))
        r_foot = (int(r_ankle[0] + 25), int(r_ankle[1] + 5))

        l_knee = (int(l_hip[0] + 35), int(l_hip[1] + 85))
        l_ankle = (int(l_knee[0] + 20), int(l_knee[1] + 80))
        l_foot = (int(l_ankle[0] + 30), int(l_ankle[1] + 5)) # Open foot ~65 deg

        # Horizontal arms & cross-bat
        l_elbow = (int(l_sh[0] + 50 * np.cos(rot_rad + 0.5)), int(l_sh[1] + 45))
        l_wrist = (int(l_elbow[0] + 55 * np.cos(rot_rad + 0.8)), int(l_elbow[1] + 10))
        r_elbow = (int(r_sh[0] + 40), int(r_sh[1] + 45))
        r_wrist = (int(l_wrist[0] + 10), int(l_wrist[1] - 5))

        bat_rad = np.radians(bat_angle)
        bat_end = (int(l_wrist[0] + 85 * np.cos(bat_rad)), int(l_wrist[1] + 85 * np.sin(bat_rad)))

    # Draw human batter anatomy with realistic widths
    skin_color = (165, 200, 235)
    kit_color = (235, 240, 245) # White cricket flannels
    pad_color = (220, 225, 230)
    helmet_color = (40, 70, 110)

    # Torso
    pts_torso = np.array([l_sh, r_sh, r_hip, l_hip], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts_torso, kit_color)
    cv2.polylines(frame, [pts_torso], True, (160, 170, 180), 2)

    # Head & Helmet
    cv2.circle(frame, nose, 18, skin_color, -1)
    cv2.circle(frame, (nose[0], nose[1] - 4), 20, helmet_color, -1)
    cv2.rectangle(frame, (nose[0] - 6, nose[1] + 2), (nose[0] + 12, nose[1] + 12), (180, 180, 180), 2) # Grille

    # Legs & Pads
    for (hip, knee, ankle, foot) in [(r_hip, r_knee, r_ankle, r_foot), (l_hip, l_knee, l_ankle, l_foot)]:
        cv2.line(frame, hip, knee, kit_color, 24)
        cv2.line(frame, knee, ankle, pad_color, 26)
        cv2.line(frame, ankle, foot, (245, 245, 245), 16) # Cricket boots

    # Arms
    cv2.line(frame, r_sh, r_elbow, kit_color, 16)
    cv2.line(frame, r_elbow, r_wrist, skin_color, 14)
    cv2.line(frame, l_sh, l_elbow, kit_color, 16)
    cv2.line(frame, l_elbow, l_wrist, skin_color, 14)

    # Bat Handle & Blade
    cv2.line(frame, l_wrist, (int((l_wrist[0] + bat_end[0]) / 2), int((l_wrist[1] + bat_end[1]) / 2)), (20, 20, 20), 7) # Handle
    cv2.line(frame, (int((l_wrist[0] + bat_end[0]) / 2), int((l_wrist[1] + bat_end[1]) / 2)), bat_end, (90, 145, 200), 16) # English Willow blade

    # Ball trajectory
    if 0.20 <= t_norm <= 0.70:
        ball_progress = (t_norm - 0.20) / 0.50
        if shot_type == "straight_drive":
            # Ball comes full and straight
            ball_x = int(640 - 20 * ball_progress)
            ball_y = int(320 + 260 * ball_progress)
            ball_r = int(5 + 7 * ball_progress)
        else:
            # Short pitched rising ball for pull shot
            ball_x = int(600 - 30 * ball_progress)
            ball_y = int(280 + 170 * ball_progress)
            ball_r = int(5 + 8 * ball_progress)
        cv2.circle(frame, (ball_x, ball_y), ball_r, (40, 40, 210), -1) # Red leather ball
        cv2.circle(frame, (ball_x, ball_y), ball_r, (255, 255, 255), 1) # Seam

    return frame

def generate_reference_set(shot_type, out_dir, count=10):
    os.makedirs(out_dir, exist_ok=True)
    w, h = 1280, 720
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    prefix = "straight" if shot_type == "straight_drive" else "pull"
    for i in range(1, count + 1):
        filename = f"{prefix}_{i:04d}.avi"
        filepath = os.path.join(out_dir, filename)
        num_frames = 50 + (i % 5) * 2 # Slight frame count variation (50-58 frames)

        writer = cv2.VideoWriter(filepath, fourcc, float(fps), (w, h))
        for f_idx in range(num_frames):
            t_norm = f_idx / float(num_frames - 1)
            frame = render_batter_frame(w, h, shot_type, t_norm, var_seed=i * 17 + f_idx)
            writer.write(frame)
        writer.release()
        print(f"Generated {filepath} ({num_frames} frames)")

if __name__ == "__main__":
    generate_reference_set("straight_drive", "data/reference_videos/straight_drive", 10)
    generate_reference_set("pull_shot", "data/reference_videos/pull_shot", 10)
    print("Multi-shot reference video generation complete.")
