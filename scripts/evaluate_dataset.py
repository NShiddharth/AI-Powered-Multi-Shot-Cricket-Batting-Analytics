import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from src.shot_analysis.shot_classifier import ShotClassifier, FEATURE_NAMES

def main():
    sc = ShotClassifier()
    classes = ['cover_drive', 'straight_drive', 'pull_shot']

    results = {c: {'correct': 0, 'total': 0, 'pred': {}} for c in classes}
    y_true = []
    y_pred = []

    for c in classes:
        json_path = f"output/reference_calibration/{c}/{c}_reference.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for v in data.get('per_video_data', []):
            vm = v.get('metrics', {})
            # Run inference with DEBUG flag disabled for clean bulk eval
            res = sc.predict(vm, debug=False)
            pred = res['predicted_shot']
            results[c]['total'] += 1
            results[c]['pred'][pred] = results[c]['pred'].get(pred, 0) + 1
            if pred == c:
                results[c]['correct'] += 1
            y_true.append(c)
            y_pred.append(pred)

    print("==================================================")
    print("EMPIRICAL EVALUATION ON ALL 30 REFERENCE VIDEOS")
    print("==================================================")
    for c in classes:
        corr = results[c]['correct']
        tot = results[c]['total']
        print(f"{c.replace('_', ' ').title()}:")
        print(f"  {tot} videos")
        print(f"  correctly classified: {corr}/{tot}")
        print(f"  predictions breakdown: {results[c]['pred']}")

    all_labels = sorted(list(set(y_true + y_pred)))
    print("\nConfusion Matrix (Rows = Ground Truth, Columns = Predicted):")
    header = f"{'Ground Truth':<16} | " + " | ".join([f"{p:>14}" for p in all_labels])
    print(header)
    print("-" * len(header))
    for t in classes:
        counts = [str(sum(1 for yt, yp in zip(y_true, y_pred) if yt == t and yp == p)) for p in all_labels]
        print(f"{t:<16} | " + " | ".join([f"{cnt:>14}" for cnt in counts]))

    # Accuracy, Precision, Recall, F1 for known classes
    total_samples = len(y_true)
    correct_samples = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct_samples / total_samples

    print("\n--------------------------------------------------")
    print(f"Overall Accuracy: {accuracy * 100:.1f}% ({correct_samples}/{total_samples})")
    print("--------------------------------------------------")

    print("\nPer-Class Performance (excluding UNKNOWN from precision/recall denominator):")
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{c.replace('_', ' ').title():<16}: Precision={prec*100:.1f}%, Recall={rec*100:.1f}%, F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})")

if __name__ == "__main__":
    main()
