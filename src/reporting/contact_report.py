"""
AthleteRise - Bat-Ball Contact Visual & Textual Reporter
Formats contact estimation diagnostics, spatial proximity, and trajectory deflection.
"""

from typing import Dict, Any

class ContactReporter:
    """
    Generates structured textual summaries and diagnostics for Bat-Ball Contact Detection.
    """
    @staticmethod
    def format_contact_summary(contact_data: Dict[str, Any]) -> str:
        status = contact_data.get('status', 'unknown').upper().replace('_', ' ')
        conf = contact_data.get('confidence', 0.0)
        est_frame = contact_data.get('estimated_frame')
        pose_frame = contact_data.get('pose_impact_frame')
        diff = contact_data.get('frame_difference')
        reason = contact_data.get('reason', '')
        signals = contact_data.get('signals', {})

        lines = [
            "Bat-Ball Contact Analysis",
            "-------------------------",
            f"Status:                    {status}",
            f"Contact Confidence:        {conf}%",
            f"Estimated Contact Frame:   {est_frame if est_frame is not None else 'N/A'}",
            f"Pose Impact Frame:         {pose_frame if pose_frame is not None else 'N/A'}",
            f"Contact Frame Difference:  {f'{diff:+d} frames' if diff is not None else 'N/A'}",
            f"Assessment:                {reason}"
        ]

        if signals:
            lines.append("Signal Breakdown:")
            if 'min_distance_px' in signals:
                lines.append(f"  - Min Bat-Ball Proximity: {signals['min_distance_px']} px")
            if 'trajectory_deflection_deg' in signals:
                lines.append(f"  - Trajectory Deflection:  {signals['trajectory_deflection_deg']}°")
            if 'spatial_score' in signals:
                lines.append(f"  - Spatial Alignment:      {signals['spatial_score']}/100")
            if 'trajectory_score' in signals:
                lines.append(f"  - Trajectory Change:      {signals['trajectory_score']}/100")

        return "\n".join(lines)
