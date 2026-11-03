"""
AthleteRise - Bat-Ball Contact Ground-Truth Validator
Provides manual validation storage and evaluation metrics (Precision, Recall, F1, TP, FP, TN, FN)
for the visual contact detector.
Annotations stored in: data/contact_annotations/
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

ANNOTATIONS_DIR = Path("data/contact_annotations")

class ContactValidator:
    """
    Manages contact ground-truth annotations and calculates detector evaluation metrics.
    """
    def __init__(self, annotations_dir: Optional[str] = None):
        self.annotations_dir = Path(annotations_dir or ANNOTATIONS_DIR)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def save_annotation(self, video_id: str, candidate_frame: int, decision: str, notes: str = "") -> str:
        """
        Saves manual validation decision: 'confirm_contact', 'no_contact', 'uncertain'.
        """
        data = {
            'video_id': video_id,
            'candidate_frame': candidate_frame,
            'decision': decision.lower(),
            'notes': notes
        }
        file_path = self.annotations_dir / f"{video_id}_contact.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return str(file_path)

    def load_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Loads all ground-truth annotations from data/contact_annotations/"""
        annotations = {}
        for fpath in glob.glob(str(self.annotations_dir / "*_contact.json")):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                vid = d.get('video_id', Path(fpath).stem.replace('_contact', ''))
                annotations[vid] = d
            except Exception:
                continue
        return annotations

    def evaluate_detector(self, detector_results: Dict[str, str]) -> Dict[str, Any]:
        """
        Compares detector predictions to ground-truth annotations.
        detector_results: {video_id: 'contact' | 'no_contact' | 'unknown'}
        """
        annotations = self.load_annotations()
        if not annotations:
            return {
                'status': 'NOT YET VALIDATED',
                'total_annotations': 0,
                'message': 'No ground-truth contact annotations found in data/contact_annotations/'
            }

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        uncertain = 0

        for vid, gt in annotations.items():
            gt_dec = gt.get('decision', '').lower()
            if gt_dec not in ('confirm_contact', 'contact', 'no_contact'):
                uncertain += 1
                continue

            pred = detector_results.get(vid, 'unknown').lower()
            is_gt_contact = gt_dec in ('confirm_contact', 'contact')
            is_pred_contact = pred in ('contact', 'contact_detected')

            if is_gt_contact and is_pred_contact:
                tp += 1
            elif not is_gt_contact and is_pred_contact:
                fp += 1
            elif not is_gt_contact and not is_pred_contact:
                tn += 1
            elif is_gt_contact and not is_pred_contact:
                fn += 1

        total = tp + fp + tn + fn
        if total == 0:
            return {
                'status': 'NOT YET VALIDATED',
                'total_annotations': len(annotations),
                'message': 'No definitive contact/no_contact ground-truth samples available.'
            }

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        accuracy = round((tp + tn) / total, 4)

        return {
            'status': 'VALIDATED',
            'total_evaluated': total,
            'uncertain_samples': uncertain,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }
