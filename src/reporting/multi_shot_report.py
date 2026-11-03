"""
AthleteRise - Multi-Shot Report Generator
Generates comprehensive text reports and JSON payloads matching
the unified architecture specifications.
"""

from typing import Dict, Any
from src.shot_analysis.shot_config import SHOT_DISPLAY_NAMES

class MultiShotReportGenerator:
    """
    Generates structured multi-shot evaluation reports.
    """
    @staticmethod
    def generate_text_report(eval_data: Dict[str, Any]) -> str:
        shot_analysis = eval_data.get('shot_analysis', {})
        shot_code = shot_analysis.get('predicted_shot', 'cover_drive')
        shot_name = SHOT_DISPLAY_NAMES.get(shot_code, shot_code.replace('_', ' ').title())
        shot_conf = shot_analysis.get('confidence', 0.0)
        conf_level = shot_analysis.get('confidence_level', 'Moderate Confidence' if shot_code != 'unknown' else 'Low Confidence')
        top_cand = shot_analysis.get('top_candidate', shot_code)
        top_cand_name = SHOT_DISPLAY_NAMES.get(top_cand, top_cand.replace('_', ' ').title())
        top_prob = shot_analysis.get('top_probability', shot_conf)
        top_margin = shot_analysis.get('top_class_margin', 0.0)
        ref_sim = shot_analysis.get('reference_similarity', 0.0)
        feat_cov = shot_analysis.get('feature_coverage', 1.0)
        shot_reason = shot_analysis.get('reason', '')

        contact = eval_data.get('contact_analysis', {})
        contact_status = contact.get('status', 'UNKNOWN').replace('_', ' ').title()
        evidence_score = contact.get('evidence_score', contact.get('confidence', 41.0))
        contact_frame = contact.get('visual_contact_frame', contact.get('contact_frame'))
        pose_frame = contact.get('biomechanical_impact_frame', contact.get('pose_impact_frame'))
        frame_diff = contact.get('frame_difference')
        contact_reason = contact.get('reason', '')

        bat_trk = eval_data.get('bat_tracking', {})
        ball_trk = eval_data.get('ball_tracking', {})

        overall_score = eval_data.get('overall_score')
        score_disp = f"{overall_score} / 100" if overall_score is not None else "N/A"
        grade = eval_data.get('skill_grade', 'N/A')

        categories = eval_data.get('category_scores', {})
        shot_specific = eval_data.get('shot_specific_metrics', {})
        feedback = eval_data.get('coaching_feedback', {})

        lines = [
            "============================================================",
            "              ATHLETERISE — BATTING ANALYSIS               ",
            "============================================================",
            "",
            "Shot Classification Summary",
            "---------------------------",
            f"Detected Shot:         {shot_name}",
            f"Classification Status: {conf_level}",
            f"Top Candidate:         {top_cand_name} ({round(top_prob * 100, 1)}%)",
            f"Top-Class Margin:      {round(top_margin * 100, 1)} percentage points",
            f"Reference Similarity:  {round(ref_sim * 100, 1)}%",
            f"Feature Coverage:      {int(feat_cov * 100)}%",
            f"Decision Reason:       {shot_reason}",
            ""
        ]

        # Probabilities breakdown
        class_probs = shot_analysis.get('class_probabilities', {})
        if class_probs:
            lines.append("Class Probabilities:")
            for sc, prob in class_probs.items():
                s_title = SHOT_DISPLAY_NAMES.get(sc, sc.replace('_', ' ').title())
                lines.append(f"  - {s_title:<16}: {round(prob * 100, 1)}%")
            lines.append("")

        # Bat & Ball Tracking Quality
        lines.extend([
            "Tracking Quality Diagnostics",
            "-----------------------------",
            f"Bat Tracking:  {bat_trk.get('status', 'AVAILABLE')} (Coverage: {bat_trk.get('coverage_pct', 0)}%, Continuity: {bat_trk.get('continuity', 'Good')})",
            f"Ball Tracking: {ball_trk.get('status', 'UNAVAILABLE')} (Coverage: {ball_trk.get('coverage_pct', 0)}%, Continuity: {ball_trk.get('continuity', 'Poor')})",
            ""
        ])

        # Contact & Impact Analysis
        lines.extend([
            "Bat–Ball Contact Analysis",
            "-------------------------",
            f"Visual Contact Status:       {contact_status}",
            f"Contact Evidence Score:      {evidence_score} / 100",
            f"Visual Contact Frame:        {f'Frame {contact_frame}' if contact_frame is not None else 'N/A'}",
            f"Biomechanical Impact Frame:  Frame {pose_frame}" if pose_frame is not None else "Biomechanical Impact Frame:  N/A",
            f"Frame Difference:            {f'{frame_diff:+d} frames' if frame_diff is not None else 'N/A'}",
            f"Contact Assessment Reason:   {contact_reason}",
            ""
        ])

        # General Biomechanical Analysis (Always Available)
        lines.append("General Biomechanical Analysis (Monocular Pose Tracking)")
        lines.append("---------------------------------------------------------")
        metric_evals = eval_data.get('metric_evaluations', {})
        if not metric_evals:
            metric_evals = eval_data.get('metric_scores', {})
        if metric_evals:
            for mk, mv in metric_evals.items():
                m_name = mv.get('name', mk.replace('_', ' ').title())
                u_val = mv.get('user_value_display', f"{mv.get('raw_value')} {mv.get('unit', '')}" if mv.get('raw_value') is not None else 'N/A')
                sc_disp = mv.get('metric_score_display', f"{mv.get('score')} / 100" if mv.get('score') is not None else 'N/A')
                assess = mv.get('assessment', '')
                lines.append(f"  * {m_name:<28}: {u_val:<12} (Score: {sc_disp}) [{assess}]")
        else:
            lines.append("No biomechanical metrics available.")
        lines.append("")

        if shot_code == 'unknown':
            score_disp = "Not Available (Shot Classification Required)"
            grade = "Classification Required"

        lines.extend([
            "Overall Technique Score",
            "-----------------------",
            f"{score_disp}",
            "",
            "Skill Grade",
            "-----------",
            f"{grade}",
            ""
        ])

        # Category Breakdown
        lines.append("Category Breakdown")
        lines.append("------------------")
        if shot_code == 'unknown':
            lines.append("Shot-specific category scoring unavailable for unclassified stroke.")
        elif categories:
            for ck, cv in categories.items():
                cat_name = cv.get('category_name', ck.replace('_', ' ').title())
                c_score = cv.get('score_display', f"{cv.get('score')} / 100")
                lines.append(f"  - {cat_name:<18}: {c_score}")
        lines.append("")

        # Coaching Feedback
        lines.append("Coaching Insights & Recommendations")
        lines.append("------------------------------------")
        strengths = feedback.get('key_strengths', [])
        if strengths and shot_code != 'unknown':
            lines.append("Key Strengths:")
            for s in strengths:
                lines.append(f"  + {s}")
        recs = feedback.get('coaching_recommendations', [])
        if recs:
            lines.append("Recommendations:")
            for r in recs:
                lines.append(f"  * {r}")
        lines.append("")
        lines.append("============================================================")

        return "\n".join(lines)
