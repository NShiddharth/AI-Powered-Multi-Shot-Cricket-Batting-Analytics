"""
AthleteRise - Cover Drive Evaluation & Scoring Engine
Implements Continuous Distribution-Based Scoring using Calibrated Reference Descriptors,
Hierarchical Category Weighting, Dynamic N/A Renormalization, and Fallback Expert Rules.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple

METRIC_POLICIES = {
    'front_foot_direction': {
        'policy': 'two_sided_optimal',
        'display_name': 'Front Foot Direction (°)',
        'description': 'Angle of front foot toward the cover region (~140°-175°). Two-sided optimal range.'
    },
    'stride_length': {
        'policy': 'two_sided_optimal',
        'display_name': 'Stride Length (norm)',
        'description': 'Forward stride length normalized by leg length. Two-sided optimal range.'
    },
    'head_knee_alignment': {
        'policy': 'lower_is_better',
        'display_name': 'Head-Knee Alignment (norm)',
        'description': 'Horizontal head offset from front knee normalized by torso length. Lower offset is better.'
    },
    'head_stability': {
        'policy': 'lower_is_better',
        'display_name': 'Head Stability (norm)',
        'description': 'Spatial displacement/variance of head during downswing. Lower displacement is better.'
    },
    'front_elbow_angle': {
        'policy': 'higher_is_better',
        'display_name': 'Front Elbow Angle (°)',
        'description': 'High front elbow elevation at contact. Higher elevation is better within anatomical limits.'
    },
    'shoulder_rotation': {
        'policy': 'distribution_similarity',
        'display_name': 'Shoulder Rotation (°)',
        'description': 'Shoulder uncoiling angle matching calibrated reference distribution profile.'
    },
    'hip_rotation': {
        'policy': 'distribution_similarity',
        'display_name': 'Hip Rotation (°)',
        'description': 'Hip rotation through contact matching calibrated reference distribution profile.'
    },
    'spine_lean': {
        'policy': 'lower_is_better',
        'display_name': 'Spine Lean Tilt (°)',
        'description': 'Spine tilt angle from vertical. Lower tilt indicates upright balanced posture.'
    },
    'follow_through_angle': {
        'policy': 'higher_is_better',
        'display_name': 'Follow-Through Angle (°)',
        'description': 'Elbow extension angle during follow-through phase. Higher follow-through is better.'
    },
    'follow_through_quality': {
        'policy': 'higher_is_better',
        'display_name': 'Follow-Through Quality (0-100)',
        'description': 'Overall follow-through extension completion score (0-100). Higher is better.'
    }
}

CATEGORY_METRIC_MAPPING = {
    "footwork": {
        "front_foot_direction": 0.60,
        "stride_length": 0.40
    },
    "head_position": {
        "head_knee_alignment": 0.60,
        "head_stability": 0.40
    },
    "swing_control": {
        "front_elbow_angle": 0.50,
        "shoulder_rotation": 0.25,
        "hip_rotation": 0.25
    },
    "balance": {
        "spine_lean": 0.60,
        "head_stability": 0.40
    },
    "follow_through": {
        "follow_through_angle": 0.50,
        "follow_through_quality": 0.50
    }
}

OVERALL_CATEGORY_WEIGHTS = {
    "swing_control": 0.25,
    "head_position": 0.20,
    "footwork": 0.20,
    "balance": 0.20,
    "follow_through": 0.15
}

def map_score_to_grade(score: Optional[float]) -> str:
    """
    Maps continuous 0-100 score to the Project Scoring Scale grade.
    Clearly designated as a Project Scoring Scale.
    """
    if score is None:
        return "N/A"
    if score >= 90.0:
        return "Advanced / Elite Execution"
    elif score >= 75.0:
        return "Above Average / Proficient"
    elif score >= 60.0:
        return "Intermediate"
    elif score >= 45.0:
        return "Developing"
    else:
        return "Beginner"

def score_metric_continuous(metric_key: str, user_val: Optional[float], ref_stat: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Computes a continuous score (0-100) and assessment text for a given user metric value
    against the calibrated statistical distribution descriptors.
    """
    if user_val is None:
        return None, "Metric N/A or occluded (excluded from weighting)"

    p10 = ref_stat.get('p10')
    p25 = ref_stat.get('p25')
    median = ref_stat.get('median')
    mean = ref_stat.get('mean')
    p75 = ref_stat.get('p75')
    p90 = ref_stat.get('p90')

    if any(v is None for v in [p10, p25, median, p75, p90]):
        return None, "Reference statistics incomplete"

    policy_info = METRIC_POLICIES.get(metric_key, {'policy': 'two_sided_optimal'})
    policy = policy_info['policy']

    if policy in ('two_sided_optimal', 'distribution_similarity'):
        # Core IQR [P25, P75]: 90 - 100
        if p25 <= user_val <= p75:
            if user_val == median or p75 == p25:
                score = 95.0
            elif user_val < median:
                denom = (median - p25) if (median - p25) > 0 else 1.0
                score = 90.0 + 10.0 * (user_val - p25) / denom
            else:
                denom = (p75 - median) if (p75 - median) > 0 else 1.0
                score = 100.0 - 10.0 * (user_val - median) / denom
            assessment = "Core Reference Range (P25-P75)"

        # Acceptable Lower [P10, P25): 70 - 89.9
        elif p10 <= user_val < p25:
            denom = (p25 - p10) if (p25 - p10) > 0 else 1.0
            score = 70.0 + 19.9 * (user_val - p10) / denom
            assessment = "Acceptable Reference Range (P10-P25)"

        # Acceptable Upper (P75, P90]: 70 - 89.9
        elif p75 < user_val <= p90:
            denom = (p90 - p75) if (p90 - p75) > 0 else 1.0
            score = 89.9 - 19.9 * (user_val - p75) / denom
            assessment = "Acceptable Reference Range (P75-P90)"

        # Below P10: progressive drop from 69.9 to 0
        elif user_val < p10:
            iqr = max(1.0, abs(p75 - p25))
            dist = p10 - user_val
            score = max(0.0, 70.0 - 70.0 * (dist / iqr))
            assessment = "Below Reference P10 Range"

        # Above P90: progressive drop from 69.9 to 0
        else:
            iqr = max(1.0, abs(p75 - p25))
            dist = user_val - p90
            score = max(0.0, 70.0 - 70.0 * (dist / iqr))
            assessment = "Above Reference P90 Range"

    elif policy == 'lower_is_better':
        # Used for head_knee_alignment, head_stability, spine_lean
        # Values <= median are top tier (90 - 100)
        if user_val <= median:
            if user_val <= p25:
                score = 98.0
            else:
                denom = (median - p25) if (median - p25) > 0 else 1.0
                score = 98.0 - 8.0 * (user_val - p25) / denom
            assessment = "Optimal Low Value (<= Median)"

        elif median < user_val <= p75:
            denom = (p75 - median) if (p75 - median) > 0 else 1.0
            score = 90.0 - 15.0 * (user_val - median) / denom
            assessment = "Acceptable Moderate Range (Median-P75)"

        elif p75 < user_val <= p90:
            denom = (p90 - p75) if (p90 - p75) > 0 else 1.0
            score = 75.0 - 15.0 * (user_val - p75) / denom
            assessment = "Elevated Range (P75-P90)"

        else: # user_val > p90
            iqr = max(1.0, abs(p75 - p25))
            dist = user_val - p90
            score = max(0.0, 60.0 - 60.0 * (dist / iqr))
            assessment = "High Deviation (> P90)"

    elif policy == 'higher_is_better':
        # Used for front_elbow_angle, follow_through_angle, follow_through_quality
        # Values >= median are top tier (90 - 100)
        if user_val >= median:
            if user_val >= p75:
                score = 98.0
            else:
                denom = (p75 - median) if (p75 - median) > 0 else 1.0
                score = 90.0 + 8.0 * (user_val - median) / denom
            assessment = "Optimal High Elevation (>= Median)"

        elif p25 <= user_val < median:
            denom = (median - p25) if (median - p25) > 0 else 1.0
            score = 75.0 + 15.0 * (user_val - p25) / denom
            assessment = "Acceptable Elevation Range (P25-Median)"

        elif p10 <= user_val < p25:
            denom = (p25 - p10) if (p25 - p10) > 0 else 1.0
            score = 60.0 + 15.0 * (user_val - p10) / denom
            assessment = "Low Elevation Range (P10-P25)"

        else: # user_val < p10
            iqr = max(1.0, abs(p75 - p25))
            dist = p10 - user_val
            score = max(0.0, 60.0 - 60.0 * (dist / iqr))
            assessment = "Significantly Below Reference (< P10)"

    else:
        score = 50.0
        assessment = "Standard Evaluation"

    return float(np.clip(score, 0.0, 100.0)), assessment


class CoverDriveScoringEngine:
    """
    Primary Scoring Authority for Cover Drive Biomechanical Analysis.
    Integrates Data-Driven Statistical Baseline with safe Fallback to Expert Rules.
    """

    def __init__(self, reference_path: str = "output/reference_calibration/cover_drive/cover_drive_reference.json", config_path: str = "config.json"):
        self.reference_path = reference_path
        self.config_path = config_path
        self.reference_data = self._load_reference_data()
        self.config_data = self._load_config_data()

    def _load_reference_data(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.reference_path):
            return None
        try:
            with open(self.reference_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'statistics' in data:
                sample_count = data.get('sample_count', 0)
                if sample_count >= 5:
                    return data
        except Exception:
            pass
        return None

    def _load_config_data(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "elbow_angle_threshold": 120,
            "head_knee_distance_threshold": 50,
            "foot_angle_lower_threshold": 45,
            "foot_angle_upper_threshold": 135,
            "spine_lean_threshold": 30
        }

    def is_reference_valid(self) -> bool:
        return self.reference_data is not None

    def evaluate(self, user_metrics: Dict[str, Optional[float]], handedness: str, expert_eval_raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes full evaluation:
        If reference database is valid, executes Continuous Statistical Scoring.
        Otherwise, falls back to Expert Rules.
        """
        if self.is_reference_valid():
            return self._evaluate_statistical(user_metrics, handedness)
        else:
            return self._evaluate_fallback_expert(user_metrics, handedness, expert_eval_raw)

    def _evaluate_statistical(self, user_metrics: Dict[str, Optional[float]], handedness: str) -> Dict[str, Any]:
        ref_stats = self.reference_data['statistics']
        sample_count = self.reference_data.get('sample_count', 10)
        ref_status = self.reference_data.get('reference_status', 'provisional')

        # 1. Compute Continuous Metric Scores (10 Metrics)
        metric_evaluations = {}
        for m_key, policy_meta in METRIC_POLICIES.items():
            user_val = user_metrics.get(m_key)
            stat = ref_stats.get(m_key, {})
            score, assessment = score_metric_continuous(m_key, user_val, stat)
            
            ref_mean = stat.get('mean')
            dev_from_mean = round(user_val - ref_mean, 4) if (user_val is not None and ref_mean is not None) else None

            metric_evaluations[m_key] = {
                'name': policy_meta['display_name'],
                'policy': policy_meta['policy'],
                'user_value': round(user_val, 4) if user_val is not None else None,
                'user_value_display': f"{user_val:.2f}" if user_val is not None else "N/A",
                'reference_mean': stat.get('mean'),
                'reference_median': stat.get('median'),
                'reference_p10': stat.get('p10'),
                'reference_p25': stat.get('p25'),
                'reference_p75': stat.get('p75'),
                'reference_p90': stat.get('p90'),
                'deviation_from_mean': dev_from_mean,
                'metric_score': round(score, 2) if score is not None else None,
                'metric_score_display': f"{score:.1f}/100" if score is not None else "N/A",
                'assessment': assessment,
                'is_valid': user_val is not None and score is not None
            }

        # 2. Compute 5 Category Scores with Dynamic Within-Category N/A Renormalization
        category_scores = {}
        for cat_name, metric_weights in CATEGORY_METRIC_MAPPING.items():
            valid_weights = {}
            valid_scores = {}
            for m_key, initial_w in metric_weights.items():
                m_score = metric_evaluations[m_key]['metric_score']
                if m_score is not None:
                    valid_weights[m_key] = initial_w
                    valid_scores[m_key] = m_score

            total_valid_weight = sum(valid_weights.values())
            if total_valid_weight > 0:
                # Renormalize weights
                renormalized_sum = sum(valid_scores[k] * (valid_weights[k] / total_valid_weight) for k in valid_weights)
                cat_score = round(renormalized_sum, 2)
                cat_status = "valid"
                cat_feedback = self._generate_category_feedback(cat_name, cat_score, metric_evaluations)
            else:
                cat_score = None
                cat_status = "N/A"
                cat_feedback = f"All metrics in {cat_name.replace('_', ' ').title()} were occluded or unavailable (N/A)."

            category_scores[cat_name] = {
                'category_name': cat_name.replace('_', ' ').title(),
                'score': cat_score,
                'score_display': f"{cat_score:.1f}/100" if cat_score is not None else "N/A",
                'status': cat_status,
                'feedback': cat_feedback,
                'included_metrics': list(valid_scores.keys()),
                'excluded_metrics': [k for k in metric_weights if k not in valid_scores]
            }

        # 3. Compute Overall Weighted Score with Dynamic Category Renormalization
        valid_cat_weights = {}
        for cat_name, initial_cat_w in OVERALL_CATEGORY_WEIGHTS.items():
            if category_scores[cat_name]['score'] is not None:
                valid_cat_weights[cat_name] = initial_cat_w

        total_cat_weight = sum(valid_cat_weights.values())
        if total_cat_weight > 0:
            overall_score = round(sum(
                category_scores[cat_name]['score'] * (valid_cat_weights[cat_name] / total_cat_weight)
                for cat_name in valid_cat_weights
            ), 2)
        else:
            overall_score = 0.0

        skill_grade = map_score_to_grade(overall_score if total_cat_weight > 0 else None)

        return {
            "evaluation_mode": "DATA-DRIVEN REFERENCE",
            "is_fallback": False,
            "overall_score": overall_score,
            "overall_score_display": f"{overall_score:.1f}/100",
            "skill_grade": skill_grade,
            "grading_scale_disclaimer": "Project Scoring Scale: 90-100 Elite, 75-89 Above Average, 60-74 Intermediate, 45-59 Developing, <45 Beginner",
            "batter_stance": handedness,
            "reference_sample_count": sample_count,
            "reference_status": ref_status,
            "category_scores": category_scores,
            "metric_evaluations": metric_evaluations,
            "category_weights_applied": {
                k: round(valid_cat_weights[k] / total_cat_weight, 4) if k in valid_cat_weights else 0.0
                for k in OVERALL_CATEGORY_WEIGHTS
            }
        }

    def _evaluate_fallback_expert(self, user_metrics: Dict[str, Optional[float]], handedness: str, expert_eval_raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Safe Fallback to config.json Expert Rules when reference JSON is missing or invalid.
        """
        scores_10 = {}
        if expert_eval_raw:
            for cat, values in expert_eval_raw.items():
                if isinstance(values, list) and values:
                    scores_10[cat] = float(np.mean(values) * 10.0)
                else:
                    scores_10[cat] = 0.0
        else:
            # Basic fallback calculation from user_metrics
            cfg = self.config_data
            elbow_val = user_metrics.get('front_elbow_angle')
            scores_10['swing_control'] = 8.5 if (elbow_val and elbow_val > cfg.get('elbow_angle_threshold', 120)) else 5.0

            hk_val = user_metrics.get('head_knee_alignment')
            scores_10['head_position'] = 8.0 if (hk_val is not None and hk_val < 0.3) else 5.0

            ff_val = user_metrics.get('front_foot_direction')
            scores_10['footwork'] = 8.5 if (ff_val and cfg.get('foot_angle_lower_threshold', 45) < ff_val < cfg.get('foot_angle_upper_threshold', 135)) else 5.0

            sl_val = user_metrics.get('spine_lean')
            scores_10['balance'] = 8.0 if (sl_val and sl_val < cfg.get('spine_lean_threshold', 30)) else 5.0

            ft_val = user_metrics.get('follow_through_angle')
            scores_10['follow_through'] = 8.5 if (ft_val and ft_val > 120) else (5.0 if ft_val is not None else 0.0)

        # Map 0-10 scores to 0-100 scale for category_scores
        category_scores = {}
        for cat_name, initial_w in OVERALL_CATEGORY_WEIGHTS.items():
            sc10 = scores_10.get(cat_name, 0.0)
            sc100 = round(sc10 * 10.0, 2)
            category_scores[cat_name] = {
                'category_name': cat_name.replace('_', ' ').title(),
                'score': sc100,
                'score_display': f"{sc100:.1f}/100",
                'status': 'valid',
                'feedback': f"Expert rule evaluation based on config.json thresholds.",
                'included_metrics': list(CATEGORY_METRIC_MAPPING[cat_name].keys()),
                'excluded_metrics': []
            }

        overall_score = round(sum(category_scores[c]['score'] * OVERALL_CATEGORY_WEIGHTS[c] for c in OVERALL_CATEGORY_WEIGHTS), 2)
        skill_grade = map_score_to_grade(overall_score)

        return {
            "evaluation_mode": "FALLBACK EXPERT RULES",
            "is_fallback": True,
            "overall_score": overall_score,
            "overall_score_display": f"{overall_score:.1f}/100",
            "skill_grade": skill_grade,
            "grading_scale_disclaimer": "Project Scoring Scale: 90-100 Elite, 75-89 Above Average, 60-74 Intermediate, 45-59 Developing, <45 Beginner",
            "batter_stance": handedness,
            "reference_sample_count": 0,
            "reference_status": "unavailable (using config.json)",
            "category_scores": category_scores,
            "metric_evaluations": {
                k: {
                    'name': METRIC_POLICIES.get(k, {}).get('display_name', k),
                    'user_value': round(user_metrics[k], 4) if user_metrics.get(k) is not None else None,
                    'user_value_display': f"{user_metrics[k]:.2f}" if user_metrics.get(k) is not None else "N/A",
                    'metric_score': None,
                    'metric_score_display': "N/A",
                    'assessment': "Evaluated via fallback binary expert rules"
                } for k in METRIC_POLICIES
            },
            "category_weights_applied": OVERALL_CATEGORY_WEIGHTS
        }

    def _generate_category_feedback(self, cat_name: str, cat_score: float, metric_evals: Dict[str, Any]) -> str:
        if cat_score >= 85.0:
            prefix = "Excellent execution! "
        elif cat_score >= 70.0:
            prefix = "Solid mechanics. "
        elif cat_score >= 50.0:
            prefix = "Moderate technique with room for development. "
        else:
            prefix = "Key area for technical refinement. "

        if cat_name == "swing_control":
            elbow_sc = metric_evals['front_elbow_angle']['metric_score']
            if elbow_sc and elbow_sc < 70:
                return prefix + "Focus on elevating and leading with a higher front elbow through contact."
            return prefix + "Strong fluid bat path and synchronized shoulder uncoiling."

        elif cat_name == "head_position":
            hk_sc = metric_evals['head_knee_alignment']['metric_score']
            if hk_sc and hk_sc < 70:
                return prefix + "Ensure head is pressed directly over front knee at the moment of contact."
            return prefix + "Head is well-aligned over the front knee with steady visual focus."

        elif cat_name == "footwork":
            ff_sc = metric_evals['front_foot_direction']['metric_score']
            if ff_sc and ff_sc < 70:
                return prefix + "Align front foot more accurately towards the cover region to open the swing arc."
            return prefix + "Confident stride length and precise front toe alignment."

        elif cat_name == "balance":
            sp_sc = metric_evals['spine_lean']['metric_score']
            if sp_sc and sp_sc < 70:
                return prefix + "Avoid excessive torso lean; maintain an upright core through impact."
            return prefix + "Well-balanced posture and controlled core stability."

        elif cat_name == "follow_through":
            ft_sc = metric_evals['follow_through_angle']['metric_score']
            if ft_sc and ft_sc < 70:
                return prefix + "Complete the shot with a full, high follow-through finish."
        return prefix


# Multi-Shot Extensions
from src.evaluation.base_scoring_engine import MultiShotScoringEngine
from src.evaluation.straight_drive_scoring import StraightDriveScoringEngine
from src.evaluation.pull_shot_scoring import PullShotScoringEngine
from src.evaluation.feedback_engine import FeedbackEngine

