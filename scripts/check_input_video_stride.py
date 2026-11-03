import sys
sys.path.insert(0, '.')
from src.preprocessing.video_preprocessing import VideoPreprocessor
from src.pose_estimation.posture_tracker import VisionPostureTracker
import numpy as np

p = VideoPreprocessor('input_video.avi')
tracker = VisionPostureTracker()
tracker.profile_video('input_video.avi')
fms = []
for idx, f in p.read_frames():
    lm = tracker.process_frame(f, idx, p.total_frames)
    if lm:
        la = lm['left_ankle']
        ra = lm['right_ankle']
        lh = lm['left_hip']
        dist = np.linalg.norm(np.array(la[:2]) - np.array(ra[:2]))
        leg = np.linalg.norm(np.array(lh[:2]) - np.array(la[:2]))
        fms.append((idx, dist/leg, la[2], ra[2]))
p.release(); tracker.close()

print(f"Total frames: {len(fms)}")
for idx, s, lv, rv in fms[::5]:
    print(f"F{idx:2d}: stride={s:.3f}, l_vis={lv:.2f}, r_vis={rv:.2f}")
print(f"Max stride: {max(s for _, s, _, _ in fms):.3f} at frame {max(fms, key=lambda x: x[1])[0]}")
