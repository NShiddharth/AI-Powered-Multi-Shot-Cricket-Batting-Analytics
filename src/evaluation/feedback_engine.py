"""
AthleteRise - Multi-Shot Coaching Feedback Engine
Generates explainable coaching feedback, strengths, and areas for improvement
tailored specifically to the shot played.
"""

from typing import Dict, List, Any

class FeedbackEngine:
    """
    Generates domain-informed coaching feedback based on category scores
    and shot-specific biomechanical parameters.
    """
    @staticmethod
    def generate_feedback(shot_type: str, category_scores: Dict[str, Any], metric_scores: Dict[str, Any], shot_specific_metrics: Dict[str, Any]) -> Dict[str, Any]:
        shot = (shot_type or "cover_drive").lower()
        strengths = []
        improvements = []
        recommendations = []

        # Category-level analysis
        for cat_name, cat_data in category_scores.items():
            sc = cat_data.get('score')
            if sc is None:
                continue
            cat_title = cat_name.replace("_", " ").title()
            if sc >= 85.0:
                strengths.append(f"Excellent {cat_title} ({sc}/100) reflecting high consistency with calibrated reference athletes.")
            elif sc < 70.0:
                improvements.append(f"Suboptimal {cat_title} ({sc}/100) below reference baseline.")

        # Shot-specific coaching guidance
        if shot == "cover_drive":
            foot_dir = metric_scores.get('front_foot_direction', {}).get('raw_value')
            elbow = metric_scores.get('front_elbow_angle', {}).get('raw_value')
            if foot_dir is not None and foot_dir < 135.0:
                recommendations.append("Point front toe closer toward extra cover (~150°-170°) to open the off-side swing arc without dragging the back shoulder.")
            if elbow is not None and elbow < 85.0:
                recommendations.append("Elevate the leading left elbow at contact to achieve higher bat leverage and keep the stroke along the ground.")
            if not recommendations:
                recommendations.append("Maintain head still over the front knee throughout the contact corridor for fluid off-side strokeplay.")

        elif shot == "straight_drive":
            foot_dir = metric_scores.get('front_foot_direction', {}).get('raw_value')
            hk = metric_scores.get('head_knee_alignment', {}).get('raw_value')
            if foot_dir is not None and (foot_dir < 80.0 or foot_dir > 125.0):
                recommendations.append("Align front foot straight down the pitch toward mid-off / bowler rather than angling wide into the covers.")
            if hk is not None and hk > 0.30:
                recommendations.append("Transfer head weight directly over the bent front knee to avoid slicing across the line of delivery.")
            if not recommendations:
                recommendations.append("Maintain full vertical bat presentation into the follow-through straight down the ground.")

        elif shot == "pull_shot":
            sh_rot = metric_scores.get('shoulder_rotation', {}).get('raw_value')
            stride = metric_scores.get('stride_length', {}).get('raw_value')
            if stride is not None and stride > 0.28:
                recommendations.append("Do not over-stride forward on the pull shot; transfer weight back onto the rear leg for rotational power.")
            if sh_rot is not None and sh_rot < 125.0:
                recommendations.append("Allow hips and shoulders to uncoil fully through square leg rather than checking the follow-through.")
            if not recommendations:
                recommendations.append("Keep eyes level over the horizontal bat plane to roll wrists on contact and keep the pull along the ground.")

        elif shot == "unknown":
            return {
                'key_strengths': [],
                'areas_for_improvement': ["Shot type could not be classified confidently."],
                'coaching_recommendations': [
                    "Shot type could not be determined confidently. Re-record with the full batter, bat, and ball clearly visible and a stable camera angle to enable shot-specific coaching."
                ]
            }

        else:
            recommendations.append("Ensure clean stroke execution aligned with reference baseline.")

        return {
            'key_strengths': strengths[:3] if strengths else ["Solid foundational stance and balanced postural readiness."],
            'areas_for_improvement': improvements[:3] if improvements else ["Minor timing variations around contact corridor."],
            'coaching_recommendations': recommendations[:3]
        }
