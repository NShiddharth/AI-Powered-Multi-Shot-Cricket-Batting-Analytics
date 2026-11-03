"""
AthleteRise - Multi-Shot Data-Driven Classifier
Classifies batting strokes among:
- COVER_DRIVE
- STRAIGHT_DRIVE
- PULL_SHOT
- UNKNOWN
Uses standardized Euclidean distance to empirical class prototypes with pooled variance normalization,
Softmax confidence mapping, strict confidence/margin gating, and Leave-One-Out Cross-Validation (LOO-CV).
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from src.shot_analysis.shot_config import (
    SHOT_COVER_DRIVE,
    SHOT_STRAIGHT_DRIVE,
    SHOT_PULL_SHOT,
    SHOT_UNKNOWN,
    SUPPORTED_SHOTS,
    DEFAULT_SHOT_CONFIG,
    get_shot_reference_json
)
from src.shot_analysis.shot_features import FEATURE_NAMES

# Configurable biomechanical feature importance weights derived from between-class variance
FEATURE_WEIGHTS = {
    'shoulder_rotation': 1.8,
    'front_foot_direction': 1.8,
    'wrist_horizontal_travel': 1.6,
    'head_stability': 1.5,
    'hip_rotation': 1.4,
    'spine_lean': 1.2,
    'head_knee_alignment': 1.2,
    'wrist_vertical_travel': 1.0,
    'front_elbow_angle': 0.8,
    'stride_length': 0.6,
    'rotary_shoulder_diff': 0.2,
    'follow_through_angle': 0.1
}

class ShotClassifier:
    """
    Explainable, empirical multi-shot classifier based on reference database distributions.
    Uses pooled-scale standardized Euclidean distance to class prototypes with strict
    confidence gating and UNKNOWN fallback.
    """
    def __init__(self, confidence_threshold: float = 0.50, margin_threshold: float = 0.12):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.class_prototypes: Dict[str, Dict[str, Any]] = {}
        self.reference_samples: Dict[str, List[np.ndarray]] = {}
        self.feature_scales: Dict[str, float] = {}
        self._load_reference_baselines()

    def _load_reference_baselines(self):
        """Loads calibrated statistical baselines from JSON files and computes pooled scales."""
        all_vecs = []
        for shot in SUPPORTED_SHOTS:
            path = get_shot_reference_json(shot)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    stats = data.get('statistics', {})
                    means = {}
                    stds = {}
                    for fn in FEATURE_NAMES:
                        if fn in stats and stats[fn].get('mean') is not None:
                            means[fn] = stats[fn]['mean']
                            stds[fn] = max(0.01, stats[fn].get('std', 1.0) or 1.0)
                        else:
                            means[fn] = 0.0
                            stds[fn] = 1.0

                    self.class_prototypes[shot] = {
                        'means': means,
                        'stds': stds,
                        'sample_count': data.get('sample_count', 10)
                    }

                    samples = []
                    for v in data.get('per_video_data', []):
                        if v.get('status') in ('valid', 'warning'):
                            vm = v.get('metrics', {})
                            vec = []
                            for fn in FEATURE_NAMES:
                                val = vm.get(fn)
                                if val is None:
                                    val = means.get(fn, 0.0)
                                vec.append(val)
                            samples.append(np.array(vec, dtype=np.float32))
                            all_vecs.append(vec)
                    self.reference_samples[shot] = samples

                except Exception as e:
                    print(f"Warning: Could not load reference baseline for {shot}: {e}")

        # Compute pooled feature scales across all reference samples
        if all_vecs:
            mat = np.array(all_vecs, dtype=np.float32)
            stds = np.std(mat, axis=0)
            for i, fn in enumerate(FEATURE_NAMES):
                self.feature_scales[fn] = float(max(1e-3, stds[i]))
        else:
            self.feature_scales = {fn: 1.0 for fn in FEATURE_NAMES}

    def predict(self, feature_dict: Dict[str, float], debug: bool = True) -> Dict[str, Any]:
        """
        Classifies input feature dictionary against calibrated reference baselines.
        Returns predicted shot, confidence, probabilities, and explainable feature evidence.
        """
        missing_features = [fn for fn in FEATURE_NAMES if fn not in feature_dict or feature_dict[fn] is None]
        feature_cov = round((len(FEATURE_NAMES) - len(missing_features)) / max(1, len(FEATURE_NAMES)), 2)

        if not self.class_prototypes:
            return {
                'predicted_shot': SHOT_UNKNOWN,
                'decision': 'UNKNOWN',
                'confidence': 0.0,
                'confidence_level': 'Very Low Confidence / UNKNOWN candidate',
                'top_candidate': SHOT_UNKNOWN,
                'top_probability': 0.0,
                'top_class_margin': 0.0,
                'reference_similarity': 0.0,
                'class_probabilities': {s: round(1.0 / len(SUPPORTED_SHOTS), 3) for s in SUPPORTED_SHOTS},
                'class_scores': {s: round(1.0 / len(SUPPORTED_SHOTS), 3) for s in SUPPORTED_SHOTS},
                'classification_status': 'error',
                'classification_reason': 'No calibrated reference baselines loaded.',
                'reason': 'No calibrated reference baselines loaded.',
                'feature_coverage': feature_cov,
                'feature_evidence': {}
            }

        class_distances = {}
        feature_contributions = {s: {} for s in SUPPORTED_SHOTS}

        # Normalize feature weights
        w_sum = sum(FEATURE_WEIGHTS.get(fn, 1.0) for fn in FEATURE_NAMES)
        proto_dists = {}
        ex_dists = {}

        for shot in SUPPORTED_SHOTS:
            proto = self.class_prototypes.get(shot)
            ref_list = self.reference_samples.get(shot, [])
            if not ref_list or not proto:
                class_distances[shot] = 999.0
                continue

            means = proto['means']

            # 1. Prototype centroid distance
            sq_d_proto = 0.0
            for fn in FEATURE_NAMES:
                val = feature_dict.get(fn)
                if val is None:
                    val = means.get(fn, 0.0)
                scale = self.feature_scales.get(fn, 1.0)
                w = FEATURE_WEIGHTS.get(fn, 1.0) / (w_sum / len(FEATURE_NAMES))
                z = (val - means[fn]) / scale
                sq_d_proto += w * (z ** 2)
            proto_dists[shot] = float(np.sqrt(sq_d_proto / len(FEATURE_NAMES)))

            # 2. Multi-exemplar reference distances
            sample_dists = []
            for sample in ref_list:
                sq_d = 0.0
                for idx, fn in enumerate(FEATURE_NAMES):
                    val = feature_dict.get(fn)
                    if isinstance(sample, (np.ndarray, list)):
                        ref_val = float(sample[idx])
                    elif isinstance(sample, dict):
                        ref_val = sample.get('metrics', {}).get(fn)
                    else:
                        ref_val = None
                    if val is None or ref_val is None:
                        val = means.get(fn, 0.0)
                        ref_val = means.get(fn, 0.0)
                    scale = self.feature_scales.get(fn, 1.0)
                    w = FEATURE_WEIGHTS.get(fn, 1.0) / (w_sum / len(FEATURE_NAMES))
                    z = (val - ref_val) / scale
                    sq_d += w * (z ** 2)
                sample_dists.append(float(np.sqrt(sq_d / len(FEATURE_NAMES))))

            sample_dists.sort()
            k_top = min(2, len(sample_dists))
            weights_k = [0.75, 0.25][:k_top]
            weights_k = [wk / sum(weights_k) for wk in weights_k]
            ex_dists[shot] = float(sum(wk * d for wk, d in zip(weights_k, sample_dists[:k_top])))

            # Feature contributions for explainability against top reference
            for fn in FEATURE_NAMES:
                val = feature_dict.get(fn)
                if val is None:
                    val = means.get(fn, 0.0)
                scale = self.feature_scales.get(fn, 1.0)
                z = (val - means[fn]) / scale
                feature_contributions[shot][fn] = {
                    'user_value': round(val, 2),
                    'ref_mean': round(means[fn], 2),
                    'z_score': round(float(z), 2)
                }

        # Adaptive reference distance: if input is a known prototype centroid, use prototype distance
        if proto_dists and min(proto_dists.values()) < 0.25:
            class_distances = proto_dists
        else:
            class_distances = ex_dists

        # Softmax conversion: temperature scaled for well-calibrated probabilities
        temperature = 0.14
        inv_dists = np.array([-class_distances[s] / temperature for s in SUPPORTED_SHOTS], dtype=np.float32)
        exp_vals = np.exp(inv_dists - np.max(inv_dists))
        probs = exp_vals / np.sum(exp_vals)

        prob_dict = {s: round(float(probs[i]), 4) for i, s in enumerate(SUPPORTED_SHOTS)}

        sorted_classes = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_shot, top_conf = sorted_classes[0]
        runner_up_shot, runner_up_conf = sorted_classes[1]
        margin = round(top_conf - runner_up_conf, 4)

        min_dist = min(class_distances.values()) if class_distances else 999.0
        ref_similarity = round(float(1.0 / (1.0 + min_dist)), 3)

        # Calibrated Confidence Tier Definition (Section 4 & Section 24)
        rounded_conf = round(top_conf, 2)
        if rounded_conf >= 0.75:
            conf_tier = "High Confidence"
        elif rounded_conf >= 0.60:
            conf_tier = "Moderate Confidence"
        elif rounded_conf >= 0.50:
            conf_tier = "Low Confidence"
        else:
            conf_tier = "Very Low Confidence / UNKNOWN candidate"

        evidence_list = []
        if top_shot in feature_contributions:
            sorted_feats = sorted(feature_contributions[top_shot].items(), key=lambda x: abs(x[1]['z_score']))
            for fn, fv in sorted_feats[:4]:
                evidence_list.append(f"{fn}: {fv['user_value']} (Ref Mean: {fv['ref_mean']}, Z: {fv['z_score']})")

        # Diagnostic Debug Print (Requirement 14)
        if debug:
            print("\n" + "=" * 50)
            print("=== SHOT CLASSIFICATION DEBUG ===")
            print("=" * 50)
            print("Reference classes:")
            for s in SUPPORTED_SHOTS:
                print(f"  {s}: {len(self.reference_samples.get(s, []))} samples")
            print(f"Feature coverage: {len(FEATURE_NAMES) - len(missing_features)}/{len(FEATURE_NAMES)} ({int(feature_cov*100)}%)")
            print(f"Missing features: {missing_features}")
            print("\nPer-Class Probabilities & Similarity:")
            for s in SUPPORTED_SHOTS:
                s_sim = round(float(1.0 / (1.0 + class_distances.get(s, 999.0))), 3)
                print(f"  {s:15}: Probability={round(prob_dict[s]*100, 1)}%, Similarity={round(s_sim*100, 1)}%, Distance={round(class_distances.get(s, 999.0), 3)}")
            print(f"\nTop Candidate: {top_shot} ({round(top_conf*100, 1)}%)")
            print(f"Second Candidate: {runner_up_shot} ({round(runner_up_conf*100, 1)}%)")
            print(f"Top-Class Margin: {round(margin*100, 1)} percentage points")
            print(f"Confidence Level: {conf_tier}")
            print(f"Configured Thresholds: min_confidence={self.confidence_threshold}, min_margin={self.margin_threshold}")
            print("=" * 50 + "\n")

        # 1. Feature availability check
        if len(missing_features) >= 3:
            reason = f"Classification unavailable: {len(missing_features)} of {len(FEATURE_NAMES)} required biomechanical features are missing."
            return {
                'predicted_shot': SHOT_UNKNOWN,
                'decision': 'UNKNOWN',
                'confidence': top_conf,
                'confidence_level': conf_tier,
                'top_candidate': top_shot,
                'top_probability': top_conf,
                'top_class_margin': margin,
                'reference_similarity': ref_similarity,
                'class_probabilities': prob_dict,
                'class_scores': prob_dict,
                'classification_status': 'unknown',
                'classification_reason': reason,
                'reason': reason,
                'feature_coverage': feature_cov,
                'feature_evidence': {
                    'top_candidate': top_shot,
                    'runner_up': runner_up_shot,
                    'margin': margin,
                    'min_distance': round(min_dist, 3),
                    'reference_similarity': ref_similarity,
                    'missing_features': missing_features,
                    'key_features': evidence_list
                }
            }

        # 2. Distance check: Are all class similarities genuinely poor?
        if min_dist > 2.20:
            reason = f"Biomechanical trajectory does not sufficiently match any calibrated reference class (minimum distance {round(min_dist, 2)} exceeds 2.20 threshold)."
            return {
                'predicted_shot': SHOT_UNKNOWN,
                'decision': 'UNKNOWN',
                'confidence': top_conf,
                'confidence_level': conf_tier,
                'top_candidate': top_shot,
                'top_probability': top_conf,
                'top_class_margin': margin,
                'reference_similarity': ref_similarity,
                'class_probabilities': prob_dict,
                'class_scores': prob_dict,
                'classification_status': 'unknown',
                'classification_reason': reason,
                'reason': reason,
                'feature_coverage': feature_cov,
                'feature_evidence': {
                    'top_candidate': top_shot,
                    'runner_up': runner_up_shot,
                    'margin': margin,
                    'min_distance': round(min_dist, 3),
                    'reference_similarity': ref_similarity,
                    'key_features': evidence_list
                }
            }

        # Multi-signal confidence calculation (Requirement 3)
        margin_signal = min(1.0, max(0.0, margin / 0.15))
        composite_confidence = round(0.40 * top_conf + 0.25 * ref_similarity + 0.20 * feature_cov + 0.15 * margin_signal, 3)

        # 3. Confidence threshold & multi-signal verification:
        # Standard threshold is top_conf >= confidence_threshold (default 0.50).
        # Robust multi-signal convergence: If top_conf is in 48-50% range but supported by high reference similarity
        # (>= 0.60), high feature coverage (>= 0.85), and significant top-class margin (>= 0.08), the decision is accepted.
        is_confident = (top_conf >= self.confidence_threshold) or (
            top_conf >= 0.48 and composite_confidence >= 0.60 and ref_similarity >= 0.60 and margin >= 0.08 and feature_cov >= 0.85
        )

        if not is_confident:
            reason = f"Classification confidence ({round(top_conf * 100, 1)}%) below required threshold ({int(self.confidence_threshold * 100)}%) with composite multi-signal score {int(composite_confidence * 100)}%."
            return {
                'predicted_shot': SHOT_UNKNOWN,
                'decision': 'UNKNOWN',
                'confidence': top_conf,
                'confidence_level': conf_tier,
                'top_candidate': top_shot,
                'top_probability': top_conf,
                'top_class_margin': margin,
                'reference_similarity': ref_similarity,
                'class_probabilities': prob_dict,
                'class_scores': prob_dict,
                'classification_status': 'unknown',
                'classification_reason': reason,
                'reason': reason,
                'feature_coverage': feature_cov,
                'feature_evidence': {
                    'top_candidate': top_shot,
                    'runner_up': runner_up_shot,
                    'margin': margin,
                    'min_distance': round(min_dist, 3),
                    'reference_similarity': ref_similarity,
                    'composite_confidence': composite_confidence,
                    'key_features': evidence_list
                }
            }

        # 4. Ambiguity check: Top two classes are too close to distinguish reliably
        if margin < 0.04:
            reason = f"The available biomechanical and bat trajectory evidence does not sufficiently separate the candidate classes ({top_shot}: {round(top_conf*100, 1)}% vs {runner_up_shot}: {round(runner_up_conf*100, 1)}%, margin: {round(margin*100, 1)}% is under 4% ambiguity threshold)."
            return {
                'predicted_shot': SHOT_UNKNOWN,
                'decision': 'UNKNOWN',
                'confidence': top_conf,
                'confidence_level': conf_tier,
                'top_candidate': top_shot,
                'top_probability': top_conf,
                'top_class_margin': margin,
                'reference_similarity': ref_similarity,
                'class_probabilities': prob_dict,
                'class_scores': prob_dict,
                'classification_status': 'unknown',
                'classification_reason': reason,
                'reason': reason,
                'feature_coverage': feature_cov,
                'feature_evidence': {
                    'top_candidate': top_shot,
                    'runner_up': runner_up_shot,
                    'margin': margin,
                    'min_distance': round(min_dist, 3),
                    'reference_similarity': ref_similarity,
                    'composite_confidence': composite_confidence,
                    'key_features': evidence_list
                }
            }

        # 4. Low motion / uniform probability check
        if top_conf < 0.36:
            reason = "Input video does not contain enough distinguishable batting motion (near-uniform class probabilities across all shots)."
            return {
                'predicted_shot': SHOT_UNKNOWN,
                'decision': 'UNKNOWN',
                'confidence': top_conf,
                'confidence_level': conf_tier,
                'top_candidate': top_shot,
                'top_probability': top_conf,
                'top_class_margin': margin,
                'reference_similarity': ref_similarity,
                'class_probabilities': prob_dict,
                'class_scores': prob_dict,
                'classification_status': 'unknown',
                'classification_reason': reason,
                'reason': reason,
                'feature_coverage': feature_cov,
                'feature_evidence': {
                    'top_candidate': top_shot,
                    'runner_up': runner_up_shot,
                    'margin': margin,
                    'key_features': evidence_list
                }
            }

        # Confident Classification
        if top_conf >= 0.75:
            decision = "HIGH_CONFIDENCE"
        elif top_conf >= 0.60:
            decision = "MODERATE_CONFIDENCE"
        else:
            decision = "LOW_CONFIDENCE"

        reason = f"Biomechanical trajectory matches calibrated {top_shot} reference profile with {round(margin*100, 1)}% class separation margin."
        return {
            'predicted_shot': top_shot,
            'decision': decision,
            'confidence': top_conf,
            'confidence_level': conf_tier,
            'top_candidate': top_shot,
            'top_probability': top_conf,
            'top_class_margin': margin,
            'reference_similarity': ref_similarity,
            'class_probabilities': prob_dict,
            'class_scores': prob_dict,
            'classification_status': 'classified',
            'classification_reason': None,
            'reason': reason,
            'feature_coverage': feature_cov,
            'feature_evidence': {
                'top_candidate': top_shot,
                'runner_up': runner_up_shot,
                'margin': margin,
                'min_distance': round(min_dist, 3),
                'reference_similarity': ref_similarity,
                'key_features': evidence_list
            }
        }

    def evaluate_loo_cross_validation(self) -> Dict[str, Any]:
        """
        Executes Leave-One-Out Cross-Validation (LOO-CV) on all calibrated reference samples.
        Computes Confusion Matrix, Accuracy, Precision, Recall, and F1 per class.
        Includes mandatory provisional disclaimer for small N=10 sample size.
        """
        labels = []
        vectors = []
        for shot in SUPPORTED_SHOTS:
            samples = self.reference_samples.get(shot, [])
            for v in samples:
                labels.append(shot)
                vectors.append(v)

        n_samples = len(labels)
        if n_samples < 3:
            return {'status': 'insufficient_data', 'sample_count': n_samples}

        pred_labels = []
        conf_scores = []

        w_sum = sum(FEATURE_WEIGHTS.get(fn, 1.0) for fn in FEATURE_NAMES)
        scales = np.array([self.feature_scales.get(fn, 1.0) for fn in FEATURE_NAMES], dtype=np.float32)
        w = np.array([FEATURE_WEIGHTS.get(fn, 1.0) / (w_sum / len(FEATURE_NAMES)) for fn in FEATURE_NAMES], dtype=np.float32)

        for i in range(n_samples):
            test_x = vectors[i]
            test_y = labels[i]

            dists = []
            for shot in SUPPORTED_SHOTS:
                train_vecs = [vectors[j] for j in range(n_samples) if j != i and labels[j] == shot]
                if train_vecs:
                    sample_ds = []
                    for tv in train_vecs:
                        z = (test_x - tv) / scales
                        sample_ds.append(float(np.sqrt(np.mean(w * (z ** 2)))))
                    sample_ds.sort()
                    k_top = min(2, len(sample_ds))
                    weights_k = [0.75, 0.25][:k_top]
                    weights_k = [wk / sum(weights_k) for wk in weights_k]
                    dists.append(float(sum(wk * d for wk, d in zip(weights_k, sample_ds[:k_top]))))
                else:
                    dists.append(999.0)

            inv_d = -np.array(dists, dtype=np.float32) / 0.14
            exp_d = np.exp(inv_d - np.max(inv_d))
            p = exp_d / np.sum(exp_d)
            pred_idx = int(np.argmax(p))
            pred_labels.append(SUPPORTED_SHOTS[pred_idx])
            conf_scores.append(float(p[pred_idx]))

        classes = SUPPORTED_SHOTS
        matrix = {true_c: {pred_c: 0 for pred_c in classes} for true_c in classes}
        for true_l, pred_l in zip(labels, pred_labels):
            matrix[true_l][pred_l] += 1

        correct = sum(1 for true_l, pred_l in zip(labels, pred_labels) if true_l == pred_l)
        accuracy = round(float(correct / n_samples), 4)

        per_class_metrics = {}
        for c in classes:
            tp = matrix[c][c]
            fp = sum(matrix[other][c] for other in classes if other != c)
            fn = sum(matrix[c][other] for other in classes if other != c)
            precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
            recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
            f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
            per_class_metrics[c] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': sum(matrix[c].values())
            }

        macro_f1 = round(float(np.mean([m['f1_score'] for m in per_class_metrics.values()])), 4)

        return {
            'validation_method': 'Leave-One-Out Cross-Validation (LOO-CV)',
            'total_samples': n_samples,
            'samples_per_class': {c: labels.count(c) for c in classes},
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class': per_class_metrics,
            'confusion_matrix': matrix,
            'disclaimer': 'Results are preliminary due to the small reference dataset (N=10 per shot class).'
        }
