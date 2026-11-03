"""
AthleteRise - Contact Annotation & Ground-Truth Manager
Stores, loads, and manages human-in-the-loop validated contact ground truth
in data/contact_annotations/ for real evaluation and model improvement.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

ANNOTATIONS_DIR = Path("data/contact_annotations")

class ContactAnnotationManager:
    """
    Manages contact annotation labels saved in data/contact_annotations/.
    Labels: 'contact', 'no_contact', 'uncertain'
    """
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or ANNOTATIONS_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_annotation(self, video_name: str, candidate_frame: int, label: str, notes: str = "") -> str:
        """
        Saves user-confirmed contact ground-truth annotation to JSON file.
        Do NOT call this silently; only when explicitly confirmed by user.
        """
        stem = Path(video_name).stem
        target_path = self.storage_dir / f"{stem}_contact_label.json"
        payload = {
            'video': os.path.basename(video_name),
            'candidate_frame': int(candidate_frame),
            'label': label.lower().strip(), # 'contact', 'no_contact', 'uncertain'
            'notes': notes
        }
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4)
        return str(target_path)

    def get_annotation(self, video_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves confirmed contact ground-truth for a video if present."""
        stem = Path(video_name).stem
        target_path = self.storage_dir / f"{stem}_contact_label.json"
        if target_path.exists():
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def list_all_annotations(self) -> List[Dict[str, Any]]:
        """Lists all stored contact annotations."""
        res = []
        for p in self.storage_dir.glob("*_contact_label.json"):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    res.append(json.load(f))
            except Exception:
                continue
        return res

    def evaluate_detector_against_annotations(self, detector_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates predictions against confirmed ground-truth annotations.
        If no ground-truth annotations exist, reports NOT YET VALIDATED.
        """
        annotations = {a['video']: a for a in self.list_all_annotations() if a.get('label') in ('contact', 'no_contact')}
        if not annotations:
            return {
                'status': 'NOT YET VALIDATED',
                'message': 'No ground-truth contact annotations exist in data/contact_annotations/ yet. Validation metrics cannot be calculated without ground truth.'
            }

        tp, tn, fp, fn = 0, 0, 0, 0
        for pred in detector_predictions:
            vname = pred.get('video')
            if vname in annotations:
                true_label = annotations[vname]['label']
                pred_status = pred.get('status')
                pred_is_contact = (pred_status == 'contact_detected')

                if true_label == 'contact' and pred_is_contact:
                    tp += 1
                elif true_label == 'no_contact' and not pred_is_contact:
                    tn += 1
                elif true_label == 'no_contact' and pred_is_contact:
                    fp += 1
                elif true_label == 'contact' and not pred_is_contact:
                    fn += 1

        total = tp + tn + fp + fn
        if total == 0:
            return {'status': 'NOT YET VALIDATED', 'message': 'No overlapping ground truth samples.'}

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        return {
            'status': 'VALIDATED',
            'sample_count': total,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }
