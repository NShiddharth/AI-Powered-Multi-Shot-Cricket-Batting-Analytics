import json

for shot in ['cover_drive', 'straight_drive', 'pull_shot']:
    path = f'output/reference_calibration/{shot}/{shot}_reference.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n{'='*25} {shot.upper()} {'='*25}")
    for k, v in data.get('statistics', {}).items():
        m = v.get('mean')
        s = v.get('std')
        p10 = v.get('p10')
        p90 = v.get('p90')
        p50 = v.get('p50')
        print(f"{k:25s} | mean: {m} | std: {s} | p10: {p10} | p50: {p50} | p90: {p90}")
