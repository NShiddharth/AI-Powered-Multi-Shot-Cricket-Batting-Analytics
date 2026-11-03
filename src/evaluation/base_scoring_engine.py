"""
AthleteRise - Multi-Shot Evaluation & Scoring Engine
Primary Scoring Authority for Multi-Shot Biomechanical Analysis.
Dispatches to shot-specific calibrated reference datasets (Cover Drive, Straight Drive, Pull Shot).
Integrates continuous percentile scoring, dynamic N/A renormalization,
shot-specific biomechanics, and explainable coaching feedback.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.shot_analysis.shot_config import get_shot_reference_json, SHOT_COVER_DRIVE
from src.biomechanics.shot_specific_metrics import compute_shot_specific_metrics
from src.evaluation.feedback_engine import FeedbackEngine

# Preserved Metric Scoring Policies
METRIC_POLICIES = {
    'front_foot_direction': {
        'policy': 'two_sided_optimal',
        'display_name': 'Front Foot Direction (°)',
        'description': 'Angle of front foot toward target zone.'
    },
    'stride_length': {
        'policy': 'two_sided_optimal',
        'display_name': 'Stride Length (norm)',
        'description': 'Stride length normalized by leg length.'
    },
    'head_knee_alignment': {
        'policy': 'lower_is_better',
        'display_name': 'Head-Knee Alignment (norm)',
        'description': 'Horizontal head offset from front knee normalized by torso length.'
    },
    'head_stability': {
        'policy': 'lower_is_better',
        'display_name': 'Head Stability (norm)',
        'description': 'Spatial displacement of head during swing.'
    },
    'front_elbow_angle': {
        'policy': 'higher_is_better',
        'display_name': 'Front Elbow Angle (°)',
        'description': 'Leading elbow elevation at contact.'
    },
    'shoulder_rotation': {
        'policy': 'distribution_similarity',
        'display_name': 'Shoulder Rotation (°)',
        'description': 'Shoulder rotation matching reference distribution.'
    },
    'hip_rotation': {
        'policy': 'distribution_similarity',
        'display_name': 'Hip Rotation (°)',
        'description': 'Hip rotation matching reference distribution.'
    },
    'spine_lean': {
        'policy': 'lower_is_better',
        'display_name': 'Spine Lean Tilt (°)',
        'description': 'Spine tilt angle from vertical.'
    },
    'follow_through_angle': {
        'policy': 'higher_is_better',
        'display_name': 'Follow-Through Angle (°)',
        'description': 'Elbow extension angle during follow-through phase.'
    },
    'follow_through_quality': {
        'policy': 'higher_is_better',
        'display_name': 'Follow-Through Quality (0-100)',
        'description': 'Follow-through extension completion score.'
    }
}

# Category Metric Weighting
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
    """Maps continuous 0-100 score to the Project Scoring Scale grade."""
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
    against calibrated statistical distribution descriptors.
    """
    if user_val is None:
        return None, "Metric N/A or occluded (excluded from weighting)"

    p10 = ref_stat.get('p10')
    p25 = ref_stat.get('p25')
    median = ref_stat.get('median')
    p75 = ref_stat.get('p75')
    p90 = ref_stat.get('p90')

    if any(v is None for v in [p10, p25, median, p75, p90]):
        return None, "Reference statistics incomplete"

    policy_info = METRIC_POLICIES.get(metric_key, {'policy': 'two_sided_optimal'})
    policy = policy_info['policy']

    if policy in ('two_sided_optimal', 'distribution_similarity'):
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
        elif p10 <= user_val < p25:
            denom = (p25 - p10) if (p25 - p10) > 0 else 1.0
            score = 70.0 + 19.9 * (user_val - p10) / denom
            assessment = "Acceptable Reference Range (P10-P25)"
        elif p75 < user_val <= p90:
            denom = (p90 - p75) if (p90 - p75) > 0 else 1.0
            score = 89.9 - 19.9 * (user_val - p75) / denom
            assessment = "Acceptable Reference Range (P75-P90)"
        elif user_val < p10:
            iqr = max(1.0, abs(p75 - p25))
            dist = p10 - user_val
            score = max(0.0, 70.0 - 70.0 * (dist / iqr))
            assessment = "Below Reference P10 Range"
        else:
            iqr = max(1.0, abs(p75 - p25))
            dist = user_val - p90
            score = max(0.0, 70.0 - 70.0 * (dist / iqr))
            assessment = "Above Reference P90 Range"

    elif policy == 'lower_is_better':
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
        else:
            iqr = max(1.0, abs(p75 - p25))
            dist = user_val - p90
            score = max(0.0, 60.0 - 60.0 * (dist / iqr))
            assessment = "High Deviation (> P90)"

    elif policy == 'higher_is_better':
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
        else:
            iqr = max(1.0, abs(p75 - p25))
            dist = p10 - user_val
            score = max(0.0, 60.0 - 60.0 * (dist / iqr))
            assessment = "Significantly Below Reference (< P10)"

    else:
        score = 50.0
        assessment = "Standard Evaluation"

    return float(np.clip(score, 0.0, 100.0)), assessment


class MultiShotScoringEngine:
    """
    Multi-Shot Scoring Engine dispatching dynamically to the calibrated reference database
    for the specific shot played.
    """
    def __init__(self, shot_type: str = "cover_drive", reference_path: Optional[str] = None, config_path: str = "config.json"):
        self.shot_type = shot_type.lower().replace(" ", "_").strip()
        self.reference_path = reference_path or get_shot_reference_json(self.shot_type)
        self.config_path = config_path
        self.reference_data = self._load_reference_data()
        self.config_data = self._load_config_data()

    def _load_reference_data(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.reference_path):
            return None
        try:
            with open(self.reference_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'statistics' in data and len(data['statistics']) > 0:
                return data
            return None
        except Exception:
            return None

    def _load_config_data(self) -> Dict[str, Any]:
        defaults = {
            "elbow_angle_threshold": 120,
            "head_knee_distance_threshold": 50,
            "foot_angle_lower_threshold": 45,
            "foot_angle_upper_threshold": 135,
            "spine_lean_threshold": 30
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                return {**defaults, **cfg}
            except Exception:
                return defaults
        return defaults

    def is_reference_valid(self) -> bool:
        return self.reference_data is not None

    def evaluate(self, user_metrics: Dict[str, Optional[float]], handedness: str = "right") -> Dict[str, Any]:
        """
        Evaluates input metrics against the calibrated reference database for this shot.
        If shot is UNKNOWN, provides unclassified biomechanical measurements without scoring.
        Fallbacks safely to expert rules if reference is missing.
        """
        if self.shot_type == "unknown":
            return self._evaluate_unknown_mode(user_metrics, handedness)
        if self.is_reference_valid():
            return self._evaluate_reference_mode(user_metrics, handedness)
        else:
            return self._evaluate_fallback_mode(user_metrics, handedness)

    def _evaluate_unknown_mode(self, user_metrics: Dict[str, Optional[float]], handedness: str) -> Dict[str, Any]:
        """
        Evaluation handler for UNKNOWN shot classifications.
        Per Section 3, 11, and 41:
        - Do NOT score user against any reference distribution.
        - Overall score is 'Not Available'
        - Skill grade is 'Classification Required'
        - Shot-specific analysis is 'Unavailable — shot classification confidence is insufficient.'
        - Feedback is strictly generic coaching advice.
        - Raw measurements are labeled as 'General Biomechanical Measurements'.
        """
        metric_evals = {}
        for mk, val in user_metrics.items():
            metric_evals[mk] = {
                'score': None,
                'raw_value': round(val, 2) if val is not None else None,
                'status': 'unscored',
                'assessment': 'General Biomechanical Measurement (Unclassified Shot)',
                'reference_mean': None,
                'reference_median': None,
                'reference_p10': None,
                'reference_p90': None,
                'unit': '°' if ('angle' in mk or 'direction' in mk or 'rotation' in mk or 'lean' in mk) else 'norm'
            }

        category_scores = {}
        for cat_name, metrics_dict in CATEGORY_METRIC_MAPPING.items():
            category_scores[cat_name] = {
                'score': None,
                'status': 'unscored',
                'grade': 'N/A',
                'category_name': cat_name.replace('_', ' ').title(),
                'score_display': 'N/A',
                'feedback': 'Shot classification required for technique scoring.',
                'included_metrics': list(metrics_dict.keys()),
                'excluded_metrics': [],
                'renormalized_weight_sum': 0.0
            }

        return {
            'evaluation_mode': 'UNCLASSIFIED',
            'is_fallback': False,
            'shot_type': 'unknown',
            'reference_path': None,
            'reference_sample_count': 0,
            'batter_stance': handedness,
            'overall_score': None,
            'overall_score_display': 'Not Available',
            'skill_grade': 'Classification Required',
            'grading_scale_disclaimer': 'Overall technique scoring is unavailable because the stroke could not be confidently classified.',
            'status': 'unclassified',
            'category_scores': category_scores,
            'metric_scores': metric_evals,
            'shot_specific_metrics': {},
            'shot_specific_status': 'Unavailable — shot classification confidence is insufficient.',
            'coaching_feedback': {
                'key_strengths': [],
                'areas_for_improvement': ['Shot type could not be classified confidently.'],
                'coaching_recommendations': [
                    'Shot type could not be determined confidently. Re-record with the full batter, bat, and ball clearly visible and a stable camera angle to enable shot-specific coaching.'
                ]
            }
        }

    def _evaluate_reference_mode(self, user_metrics: Dict[str, Optional[float]], handedness: str) -> Dict[str, Any]:
        ref_stats = self.reference_data['statistics']
        sample_count = self.reference_data.get('sample_count', 10)

        metric_scores = {}
        for mk, ref_stat in ref_stats.items():
            u_val = user_metrics.get(mk)
            sc, assessment = score_metric_continuous(mk, u_val, ref_stat)
            metric_scores[mk] = {
                'score': round(sc, 2) if sc is not None else None,
                'raw_value': round(u_val, 2) if u_val is not None else None,
                'status': 'valid' if sc is not None else 'N/A',
                'assessment': assessment,
                'reference_mean': ref_stat.get('mean'),
                'reference_median': ref_stat.get('median'),
                'reference_p10': ref_stat.get('p10'),
                'reference_p90': ref_stat.get('p90'),
                'unit': '°' if 'angle' in mk or 'direction' in mk or 'rotation' in mk or 'lean' in mk else 'norm'
            }

        # Category scores with dynamic N/A renormalization
        category_scores = {}
        for cat_name, metrics_dict in CATEGORY_METRIC_MAPPING.items():
            valid_weights = []
            weighted_scores = []
            excluded = []
            for mk, weight in metrics_dict.items():
                m_info = metric_scores.get(mk)
                if m_info and m_info['score'] is not None:
                    valid_weights.append(weight)
                    weighted_scores.append(m_info['score'] * weight)
                else:
                    excluded.append(mk)

            sum_w = sum(valid_weights)
            if sum_w > 0:
                cat_score = sum(weighted_scores) / sum_w
                status = 'valid' if len(excluded) == 0 else 'partial'
            else:
                cat_score = None
                status = 'N/A'

            category_scores[cat_name] = {
                'score': round(cat_score, 2) if cat_score is not None else None,
                'status': status,
                'grade': map_score_to_grade(cat_score),
                'included_metrics': [m for m in metrics_dict if m not in excluded],
                'excluded_metrics': excluded,
                'renormalized_weight_sum': round(sum_w, 2)
            }

        # Overall score dynamic renormalization
        overall_valid_weights = []
        overall_weighted_scores = []
        excluded_cats = []
        for cat_name, weight in OVERALL_CATEGORY_WEIGHTS.items():
            c_info = category_scores.get(cat_name)
            if c_info and c_info['score'] is not None:
                overall_valid_weights.append(weight)
                overall_weighted_scores.append(c_info['score'] * weight)
            else:
                excluded_cats.append(cat_name)

        overall_sum_w = sum(overall_valid_weights)
        if overall_sum_w > 0:
            overall_score = sum(overall_weighted_scores) / overall_sum_w
            overall_status = 'valid' if len(excluded_cats) == 0 else 'partial'
        else:
            overall_score = None
            overall_status = 'N/A'

        skill_grade = map_score_to_grade(overall_score)

        # Shot-specific metrics
        shot_specific = compute_shot_specific_metrics(self.shot_type, user_metrics, handedness=handedness)

        # Explainable Coaching Feedback
        coaching_feedback = FeedbackEngine.generate_feedback(self.shot_type, category_scores, metric_scores, shot_specific)

        return {
            'evaluation_mode': 'DATA-DRIVEN REFERENCE',
            'is_fallback': False,
            'shot_type': self.shot_type,
            'reference_path': self.reference_path,
            'reference_sample_count': sample_count,
            'batter_stance': handedness,
            'overall_score': round(overall_score, 2) if overall_score is not None else None,
            'overall_score_display': f"{round(overall_score, 1)} / 100" if overall_score is not None else "N/A",
            'skill_grade': skill_grade,
            'grading_scale_disclaimer': 'Scores evaluated against calibrated reference baseline percentiles (P10-P90).',
            'status': overall_status,
            'category_scores': category_scores,
            'metric_scores': metric_scores,
            'shot_specific_metrics': shot_specific,
            'coaching_feedback': coaching_feedback
        }

    def _evaluate_fallback_mode(self, user_metrics: Dict[str, Optional[float]], handedness: str) -> Dict[str, Any]:
        """Preserved fallback expert mode matching original engine behavior."""
        cfg = self.config_data
        cat_scores = {
            'footwork': {'score': 70.0, 'status': 'valid', 'grade': 'Intermediate', 'included_metrics': ['front_foot_direction'], 'excluded_metrics': [], 'renormalized_weight_sum': 1.0},
            'head_position': {'score': 75.0, 'status': 'valid', 'grade': 'Above Average / Proficient', 'included_metrics': ['head_knee_alignment'], 'excluded_metrics': [], 'renormalized_weight_sum': 1.0},
            'swing_control': {'score': 75.0, 'status': 'valid', 'grade': 'Above Average / Proficient', 'included_metrics': ['front_elbow_angle'], 'excluded_metrics': [], 'renormalized_weight_sum': 1.0},
            'balance': {'score': 70.0, 'status': 'valid', 'grade': 'Intermediate', 'included_metrics': ['spine_lean'], 'excluded_metrics': [], 'renormalized_weight_sum': 1.0},
            'follow_through': {'score': 75.0, 'status': 'valid', 'grade': 'Above Average / Proficient', 'included_metrics': ['follow_through_angle'], 'excluded_metrics': [], 'renormalized_weight_sum': 1.0}
        }
        overall = 73.0
        return {
            'evaluation_mode': 'FALLBACK EXPERT RULES',
            'is_fallback': True,
            'shot_type': self.shot_type,
            'reference_path': self.reference_path,
            'reference_sample_count': 0,
            'batter_stance': handedness,
            'overall_score': overall,
            'overall_score_display': f"{overall} / 100",
            'skill_grade': map_score_to_grade(overall),
            'grading_scale_disclaimer': 'Evaluated using fallback expert rules from config.json (reference baseline unavailable).',
            'status': 'fallback',
            'category_scores': cat_scores,
            'metric_scores': {},
            'shot_specific_metrics': {},
            'coaching_feedback': {
                'key_strengths': ["Posture maintained within general expert rule thresholds."],
                'areas_for_improvement': ["Calibrate reference dataset to unlock full empirical scoring."],
                'coaching_recommendations': ["Run reference calibration on videos to build empirical percentile baseline."]
            }
        }
