import sys
sys.path.insert(0, '.')
import json
import numpy as np
from src.shot_analysis.shot_config import SUPPORTED_SHOTS, get_shot_reference_json
from src.shot_analysis.shot_features import FEATURE_NAMES

# Load samples
samples = {s: [] for s in SUPPORTED_SHOTS}
for shot in SUPPORTED_SHOTS:
    p = get_shot_reference_json(shot)
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    for v in d.get('per_video_data', []):
        if v.get('status') in ('valid', 'warning'):
            m = v.get('metrics', {})
            vec = [m.get(fn, 0.0) if m.get(fn) is not None else 0.0 for fn in FEATURE_NAMES]
            samples[shot].append(vec)

labels = []
X = []
for s in SUPPORTED_SHOTS:
    for v in samples[s]:
        labels.append(s)
        X.append(v)
X = np.array(X, dtype=np.float32)

# Compute feature F-statistic / variance ratio across classes to find most discriminative features
grand_mean = np.mean(X, axis=0)
between_var = np.zeros(len(FEATURE_NAMES))
within_var = np.zeros(len(FEATURE_NAMES))

for s in SUPPORTED_SHOTS:
    mat = np.array(samples[s])
    s_mean = np.mean(mat, axis=0)
    between_var += len(mat) * (s_mean - grand_mean)**2
    within_var += np.sum((mat - s_mean)**2, axis=0)

f_scores = (between_var / 2.0) / (within_var / 27.0 + 1e-6)

print("Feature Discriminative Power (F-score):")
for fn, f in sorted(zip(FEATURE_NAMES, f_scores), key=lambda x: x[1], reverse=True):
    print(f"  {fn:25s}: {f:6.2f}")

# Test feature weighting in LOO-CV
weights = np.sqrt(np.maximum(0.2, f_scores))
weights = weights / np.sum(weights)

n_samples = len(labels)
preds = []
conf_list = []

for i in range(n_samples):
    test_x = X[i]
    test_y = labels[i]
    
    # Train on all others
    train_protos = {}
    train_all = []
    for s in SUPPORTED_SHOTS:
        train_s = [X[j] for j in range(n_samples) if j != i and labels[j] == s]
        train_protos[s] = np.mean(train_s, axis=0)
        train_all.extend(train_s)
        
    scales = np.std(train_all, axis=0) + 1e-4
    
    dists = {}
    for s in SUPPORTED_SHOTS:
        z = (test_x - train_protos[s]) / scales
        # Weighted distance
        w_d = np.sqrt(np.sum(weights * (z**2)))
        dists[s] = w_d
        
    # Softmax
    inv_d = -np.array([dists[s] for s in SUPPORTED_SHOTS]) / 0.50
    exp_d = np.exp(inv_d - np.max(inv_d))
    p = exp_d / np.sum(exp_d)
    pred_s = SUPPORTED_SHOTS[np.argmax(p)]
    preds.append(pred_s)

acc = sum(1 for y, p in zip(labels, preds) if y == p) / n_samples
print(f"\nWeighted Prototype LOO-CV Accuracy: {acc*100:.1f}%")
for s in SUPPORTED_SHOTS:
    c = sum(1 for y, p in zip(labels, preds) if y == s and p == s)
    tot = sum(1 for y in labels if y == s)
    print(f"  {s}: {c}/{tot} ({c/tot*100:.1f}%)")
