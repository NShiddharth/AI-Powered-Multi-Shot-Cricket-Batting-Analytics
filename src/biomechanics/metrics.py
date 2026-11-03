import numpy as np
import math

def calculate_angle(a, b, c):
    """
    Calculates 2D interior angle at joint b between points a-b-c in degrees (0 to 180).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

def detect_handedness(landmarks_list):
    """
    Detects whether batter is Right-Handed ('right') or Left-Handed ('left').
    Examines relative positioning of wrists, shoulders, and feet across frames.
    """
    if not landmarks_list:
        return 'right'

    left_wrist_x_diffs = []
    right_wrist_x_diffs = []

    for lm in landmarks_list:
        if lm is None:
            continue
        try:
            lw_x = lm.get('left_wrist', [0, 0])[0]
            rw_x = lm.get('right_wrist', [0, 0])[0]
            ls_x = lm.get('left_shoulder', [0, 0])[0]
            rs_x = lm.get('right_shoulder', [0, 0])[0]

            left_wrist_x_diffs.append(abs(lw_x - ls_x))
            right_wrist_x_diffs.append(abs(rw_x - rs_x))
        except Exception:
            continue

    if not left_wrist_x_diffs:
        return 'right'

    mean_left_ext = np.mean(left_wrist_x_diffs)
    mean_right_ext = np.mean(right_wrist_x_diffs)

    return 'right' if mean_left_ext >= mean_right_ext * 0.85 else 'left'

class BiomechanicsCalculator:
    """
    Calculator for the 10 core Cover Drive biomechanical metrics.
    Supports right-handed and left-handed batters.
    Applies anatomical validity checks and strict None handling for invalid/occluded keypoints.
    """
    def __init__(self, handedness='right', frame_width=640, frame_height=480):
        self.handedness = handedness
        self.frame_width = frame_width
        self.frame_height = frame_height

    def extract_frame_landmarks(self, pose_landmarks, mp_pose):
        """
        Parses MediaPipe pose landmarks into a standard coordinate dictionary
        including visibility/confidence scores.
        """
        if not pose_landmarks:
            return None

        lm = pose_landmarks.landmark

        def get_coords(idx):
            vis = getattr(lm[idx], 'visibility', 1.0)
            return [lm[idx].x, lm[idx].y, vis]

        return {
            'left_shoulder': get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            'right_shoulder': get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            'left_elbow': get_coords(mp_pose.PoseLandmark.LEFT_ELBOW.value),
            'right_elbow': get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            'left_wrist': get_coords(mp_pose.PoseLandmark.LEFT_WRIST.value),
            'right_wrist': get_coords(mp_pose.PoseLandmark.RIGHT_WRIST.value),
            'left_hip': get_coords(mp_pose.PoseLandmark.LEFT_HIP.value),
            'right_hip': get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value),
            'left_knee': get_coords(mp_pose.PoseLandmark.LEFT_KNEE.value),
            'right_knee': get_coords(mp_pose.PoseLandmark.RIGHT_KNEE.value),
            'left_ankle': get_coords(mp_pose.PoseLandmark.LEFT_ANKLE.value),
            'right_ankle': get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            'nose': get_coords(mp_pose.PoseLandmark.NOSE.value),
            'left_foot_index': get_coords(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value),
            'right_foot_index': get_coords(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value),
        }

    def compute_frame_metrics(self, lm):
        """
        Computes instantaneous metric readings for a single frame.
        Robust Anatomical Validation:
          - Spine Lean: Anatomical validity check (shoulder above hip; tilt <= 60 deg). Returns None if invalid.
          - Front Foot Direction: Landmark confidence validation (visibility >= 0.5). Returns None if occluded.
          - Head-Knee Alignment: Torso length normalized ratio.
          - Head Stability: Shoulder width normalized coords.
        """
        if lm is None:
            return None

        is_right = (self.handedness == 'right')

        def p2d(pt): return [pt[0], pt[1]]
        def vis(pt): return pt[2] if len(pt) > 2 else 1.0

        front_shoulder = p2d(lm['left_shoulder']) if is_right else p2d(lm['right_shoulder'])
        front_elbow = p2d(lm['left_elbow']) if is_right else p2d(lm['right_elbow'])
        front_wrist = p2d(lm['left_wrist']) if is_right else p2d(lm['right_wrist'])
        front_hip = p2d(lm['left_hip']) if is_right else p2d(lm['right_hip'])
        front_knee = p2d(lm['left_knee']) if is_right else p2d(lm['right_knee'])
        front_ankle = p2d(lm['left_ankle']) if is_right else p2d(lm['right_ankle'])
        front_foot = p2d(lm['left_foot_index']) if is_right else p2d(lm['right_foot_index'])

        back_ankle = p2d(lm['right_ankle']) if is_right else p2d(lm['left_ankle'])

        front_shoulder_vis = vis(lm['left_shoulder']) if is_right else vis(lm['right_shoulder'])
        front_hip_vis = vis(lm['left_hip']) if is_right else vis(lm['right_hip'])
        front_knee_vis = vis(lm['left_knee']) if is_right else vis(lm['right_knee'])
        front_ankle_vis = vis(lm['left_ankle']) if is_right else vis(lm['right_ankle'])
        front_foot_vis = vis(lm['left_foot_index']) if is_right else vis(lm['right_foot_index'])

        # 1. Front Elbow Angle (Anatomical bounds: 25 deg to 180 deg)
        front_elbow_vis = vis(lm['left_elbow']) if is_right else vis(lm['right_elbow'])
        front_wrist_vis = vis(lm['left_wrist']) if is_right else vis(lm['right_wrist'])
        if min(front_shoulder_vis, front_elbow_vis, front_wrist_vis) >= 0.35:
            raw_elbow = calculate_angle(front_shoulder, front_elbow, front_wrist)
            if 25.0 <= raw_elbow <= 180.0:
                front_elbow_angle = float(raw_elbow)
            else:
                front_elbow_angle = None
        else:
            front_elbow_angle = None

        # Rear Elbow Angle (for cross-bat vs vertical drive arm mechanics)
        rear_shoulder = p2d(lm['right_shoulder']) if is_right else p2d(lm['left_shoulder'])
        rear_elbow = p2d(lm['right_elbow']) if is_right else p2d(lm['left_elbow'])
        rear_wrist = p2d(lm['right_wrist']) if is_right else p2d(lm['left_wrist'])
        rear_shoulder_vis = vis(lm['right_shoulder']) if is_right else vis(lm['left_shoulder'])
        rear_elbow_vis = vis(lm['right_elbow']) if is_right else vis(lm['left_elbow'])
        rear_wrist_vis = vis(lm['right_wrist']) if is_right else vis(lm['left_wrist'])
        if min(rear_shoulder_vis, rear_elbow_vis, rear_wrist_vis) >= 0.35:
            raw_rear_elbow = calculate_angle(rear_shoulder, rear_elbow, rear_wrist)
            if 25.0 <= raw_rear_elbow <= 180.0:
                rear_elbow_angle = float(raw_rear_elbow)
            else:
                rear_elbow_angle = None
        else:
            rear_elbow_angle = None

        # 2. Spine Lean (Anatomical Validity & Orientation Check)
        # In 2D image coordinates (Y increases downwards), shoulder MUST be above hip (front_shoulder[1] < front_hip[1])
        if front_shoulder_vis >= 0.5 and front_hip_vis >= 0.5 and front_shoulder[1] < front_hip[1]:
            raw_spine_angle = calculate_angle(front_hip, front_shoulder, [front_shoulder[0], front_shoulder[1] - 1.0])
            spine_tilt = abs(180.0 - raw_spine_angle)
            # Biomechanical boundary check: Normal spine tilt during batting is <= 60 degrees
            if spine_tilt <= 60.0:
                spine_lean_tilt = float(spine_tilt)
            else:
                spine_lean_tilt = None
        else:
            spine_lean_tilt = None

        # 3. Head-Over-Knee Vertical Alignment (Normalized by Torso Length)
        shoulder_mid = [(p2d(lm['left_shoulder'])[0] + p2d(lm['right_shoulder'])[0]) / 2.0, (p2d(lm['left_shoulder'])[1] + p2d(lm['right_shoulder'])[1]) / 2.0]
        hip_mid = [(p2d(lm['left_hip'])[0] + p2d(lm['right_hip'])[0]) / 2.0, (p2d(lm['left_hip'])[1] + p2d(lm['right_hip'])[1]) / 2.0]
        torso_length = np.linalg.norm(np.array(shoulder_mid) - np.array(hip_mid)) + 1e-6

        head_knee_offset_x = abs(p2d(lm['nose'])[0] - front_knee[0])
        head_knee_align_norm = head_knee_offset_x / torso_length

        # 4. Front Foot Direction (Landmark confidence validation >= 0.4, returns None if occluded)
        if min(front_knee_vis, front_ankle_vis, front_foot_vis) >= 0.4:
            front_foot_dir = float(calculate_angle(front_knee, front_ankle, front_foot))
        else:
            front_foot_dir = None

        # 5. Shoulder Rotation (Consistent uncoiling orientation)
        shoulder_vector = [p2d(lm['right_shoulder'])[0] - p2d(lm['left_shoulder'])[0], p2d(lm['right_shoulder'])[1] - p2d(lm['left_shoulder'])[1]]
        raw_sh_rot = abs(np.arctan2(shoulder_vector[1], shoulder_vector[0]) * 180.0 / np.pi)
        # Ensure consistent uncoiling orientation across opposing camera/stance angles
        shoulder_rotation = 180.0 - raw_sh_rot if raw_sh_rot < 90.0 else raw_sh_rot

        # 6. Hip Rotation (Consistent uncoiling orientation)
        hip_vector = [p2d(lm['right_hip'])[0] - p2d(lm['left_hip'])[0], p2d(lm['right_hip'])[1] - p2d(lm['left_hip'])[1]]
        raw_hip_rot = abs(np.arctan2(hip_vector[1], hip_vector[0]) * 180.0 / np.pi)
        hip_rotation = 180.0 - raw_hip_rot if raw_hip_rot < 90.0 else raw_hip_rot

        # Shoulder-Hip Separation
        sh_hip_sep = abs(shoulder_rotation - hip_rotation)

        # 7. Stride Length
        ankle_dist = np.linalg.norm(np.array(front_ankle) - np.array(back_ankle))
        leg_length = np.linalg.norm(np.array(front_hip) - np.array(front_ankle)) + 1e-6
        normalized_stride_length = float(ankle_dist / leg_length)

        # 8. Shoulder Width for Head Stability Normalization
        shoulder_width = np.linalg.norm(np.array(p2d(lm['left_shoulder'])) - np.array(p2d(lm['right_shoulder']))) + 1e-6

        return {
            'front_elbow_angle': front_elbow_angle,
            'rear_elbow_angle': rear_elbow_angle,
            'spine_lean': spine_lean_tilt,
            'head_knee_alignment': float(head_knee_align_norm),
            'front_foot_direction': front_foot_dir,
            'shoulder_rotation': float(shoulder_rotation),
            'hip_rotation': float(hip_rotation),
            'shoulder_hip_separation': float(sh_hip_sep),
            'stride_length': float(normalized_stride_length),
            'nose_x_norm': float(p2d(lm['nose'])[0] / shoulder_width),
            'nose_y_norm': float(p2d(lm['nose'])[1] / shoulder_width),
            'nose_x': float(p2d(lm['nose'])[0] * self.frame_width),
            'nose_y': float(p2d(lm['nose'])[1] * self.frame_height)
        }
