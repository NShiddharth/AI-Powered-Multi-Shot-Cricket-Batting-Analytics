import sys
sys.path.insert(0, '.')
import glob
import numpy as np
from src.preprocessing.video_preprocessing import VideoPreprocessor
from src.pose_estimation.posture_tracker import VisionPostureTracker
from src.biomechanics.metrics import detect_handedness

for shot in ['cover_drive', 'straight_drive', 'pull_shot']:
    vids = sorted(glob.glob(f'data/reference_videos/{shot}/*.avi'))
    sh_angles_raw = []
    sh_angles_aligned = []
    hip_angles_aligned = []
    
    for vp in vids:
        p = VideoPreprocessor(vp)
        tracker = VisionPostureTracker()
        tracker.profile_video(vp)
        fms = []
        for idx, f in p.read_frames():
            lm = tracker.process_frame(f, idx, p.total_frames)
            if lm: fms.append(lm)
        p.release(); tracker.close()
        
        if fms:
            # Determine handedness and facing direction
            h = detect_handedness(fms)
            mid = fms[len(fms)//2]
            is_right = (h == 'right')
            fs = mid['left_shoulder'] if is_right else mid['right_shoulder']
            rs = mid['right_shoulder'] if is_right else mid['left_shoulder']
            fh = mid['left_hip'] if is_right else mid['right_hip']
            rh = mid['right_hip'] if is_right else mid['left_hip']
            
            # Forward vector from rear shoulder to front shoulder
            dx_sh = fs[0] - rs[0]
            dy_sh = fs[1] - rs[1]
            
            # Batter facing direction based on average stance
            facing_dir = -1.0 if dx_sh < 0 else 1.0
            
            # Normalized forward vector
            norm_dx_sh = dx_sh * facing_dir
            norm_dy_sh = dy_sh  # Y increases downward
            
            # Undirected line angle with horizontal (0 to 90 degrees)
            tilt_sh = abs(np.arctan2(abs(dy_sh), abs(dx_sh))) * 180.0 / np.pi
            
            # Directed angle in batter's forward frame (-180 to 180)
            dir_sh = np.arctan2(norm_dy_sh, norm_dx_sh) * 180.0 / np.pi
            
            sh_angles_raw.append(abs(np.arctan2(rs[1]-ls[1] if 'ls' in locals() else dy_sh, rs[0]-ls[0] if 'ls' in locals() else dx_sh))*180/np.pi)
            sh_angles_aligned.append(tilt_sh)
            
            dx_hip = fh[0] - rh[0]
            dy_hip = fh[1] - rh[1]
            tilt_hip = abs(np.arctan2(abs(dy_hip), abs(dx_hip))) * 180.0 / np.pi
            hip_angles_aligned.append(tilt_hip)
            
    print(f"\n{'='*15} {shot.upper()} ALIGNED TILT {'='*15}")
    print(f"Shoulder Tilt: Mean={np.mean(sh_angles_aligned):.1f}, Std={np.std(sh_angles_aligned):.1f}, P10={np.percentile(sh_angles_aligned, 10):.1f}, P90={np.percentile(sh_angles_aligned, 90):.1f}")
    print(f"Hip Tilt:      Mean={np.mean(hip_angles_aligned):.1f}, Std={np.std(hip_angles_aligned):.1f}, P10={np.percentile(hip_angles_aligned, 10):.1f}, P90={np.percentile(hip_angles_aligned, 90):.1f}")
