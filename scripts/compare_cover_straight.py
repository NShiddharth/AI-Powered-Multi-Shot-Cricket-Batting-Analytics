import numpy as np
import json
import sys
sys.path.insert(0, '.')
from src.shot_analysis.shot_features import FEATURE_NAMES

with open('output/reference_calibration/cover_drive/cover_drive_reference.json') as f:
    cov_data = json.load(f)
with open('output/reference_calibration/straight_drive/straight_drive_reference.json') as f:
    str_data = json.load(f)
with open('output/reference_calibration/pull_shot/pull_shot_reference.json') as f:
    pul_data = json.load(f)

cov_vecs = np.array([[v['metrics'][fn] for fn in FEATURE_NAMES] for v in cov_data['per_video_data']])
str_vecs = np.array([[v['metrics'][fn] for fn in FEATURE_NAMES] for v in str_data['per_video_data']])
pul_vecs = np.array([[v['metrics'][fn] for fn in FEATURE_NAMES] for v in pul_data['per_video_data']])

print("=== Cohen d Effect Sizes ===")
print(f"{'Feature':28} | {'Cov vs Str':12} | {'Cov vs Pul':12} | {'Str vs Pul':12}")
print("-" * 75)
for i, fn in enumerate(FEATURE_NAMES):
    c_m, c_s = np.mean(cov_vecs[:, i]), np.std(cov_vecs[:, i])
    s_m, s_s = np.mean(str_vecs[:, i]), np.std(str_vecs[:, i])
    p_m, p_s = np.mean(pul_vecs[:, i]), np.std(pul_vecs[:, i])

    d_cs = abs(c_m - s_m) / (np.sqrt(0.5 * (c_s**2 + s_s**2)) or 1.0)
    d_cp = abs(c_m - p_m) / (np.sqrt(0.5 * (c_s**2 + p_s**2)) or 1.0)
    d_sp = abs(s_m - p_m) / (np.sqrt(0.5 * (s_s**2 + p_s**2)) or 1.0)

    print(f"{fn:28} | {d_cs:10.2f}   | {d_cp:10.2f}   | {d_sp:10.2f}")
