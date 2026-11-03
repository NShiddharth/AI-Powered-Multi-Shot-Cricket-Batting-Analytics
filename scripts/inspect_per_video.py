import json

for shot in ['cover_drive', 'straight_drive', 'pull_shot']:
    path = f'output/reference_calibration/{shot}/{shot}_reference.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n{'='*20} {shot.upper()} PER-VIDEO METRICS {'='*20}")
    for v in data.get('per_video_data', []):
        vid = v.get('video_id')
        m = v.get('metrics', {})
        h = v.get('handedness')
        cf = v.get('contact_frame')
        fea = m.get('front_elbow_angle')
        sr = m.get('shoulder_rotation')
        sl = m.get('spine_lean')
        ffd = m.get('front_foot_direction')
        print(f"{vid:15s} (H:{h:5s}, CF:{cf:2d}): elbow={fea:6.1f}, sh_rot={sr:6.1f}, spine={sl:5.1f}, foot_dir={ffd:6.1f}")
