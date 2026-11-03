import sys
sys.path.insert(0, '.')
import json
import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.shot_analysis.shot_config import SUPPORTED_SHOTS, get_shot_reference_json
from src.shot_analysis.shot_features import FEATURE_NAMES

# Load current reference samples from JSON
data_by_shot = {}
all_labels = []
all_vectors = []

for shot in SUPPORTED_SHOTS:
    p = get_shot_reference_json(shot)
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    for v in d.get('per_video_data', []):
        if v.get('status') in ('valid', 'warning'):
            m = v.get('metrics', {})
            vec = [m.get(fn, 0.0) if m.get(fn) is not None else 0.0 for fn in FEATURE_NAMES]
            all_labels.append(shot)
            all_vectors.append(vec)

X = np.array(all_vectors, dtype=np.float32)
y = np.array(all_labels)

print(f"Dataset: X={X.shape}, y={len(y)}")

loo = LeaveOneOut()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "SVM (linear)": SVC(kernel='linear', probability=True, C=1.0),
    "SVM (rbf)": SVC(kernel='rbf', probability=True, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, max_depth=4)
}

for name, clf in models.items():
    preds = []
    for train_ix, test_ix in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_ix])
        X_test = scaler.transform(X[test_ix])
        clf.fit(X_train, y[train_ix])
        preds.append(clf.predict(X_test)[0])
    
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    print(f"\nModel: {name}")
    print(f"  LOO-CV Accuracy: {acc*100:.1f}%, Macro F1: {f1*100:.1f}%")
    print(f"  Confusion Matrix:\n{confusion_matrix(y, preds, labels=SUPPORTED_SHOTS)}")
