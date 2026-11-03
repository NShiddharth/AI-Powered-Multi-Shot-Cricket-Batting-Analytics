"""
AthleteRise - Bat-Ball Contact Estimator
Evaluates multi-signal visual evidence around the biomechanical impact window:
- Spatial distance from ball to bat line segment
- Temporal alignment around candidate contact window
- Ball trajectory deflection angle post-contact
- Relative velocity vector between ball and bat
- Bat trajectory continuity
Distinguishes POSE IMPACT ESTIMATE from VISUAL BAT-BALL CONTACT.
Enforces strict tracking availability, unified 3-state contact decisions
(CONTACT, NO_CONTACT, UNKNOWN), and eliminates logical contradictions.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from src.contact_detection.contact_features import point_to_segment_distance, compute_trajectory_deflection

STATUS_CONTACT = "contact"
STATUS_CONTACT_DETECTED = "contact"
STATUS_NO_CONTACT = "no_contact"
STATUS_UNKNOWN = "unknown"

class ContactEstimator:
    """
    Dedicated visual contact estimation engine combining spatial, temporal,
    and trajectory deflection signals.
    Enforces strict tracking availability, unified 3-state contact decisions
    (CONTACT, NO_CONTACT, UNKNOWN), and eliminates logical contradictions.
    """
    def __init__(self, spatial_threshold_px: float = 75.0, deflection_threshold_deg: float = 25.0):
        self.spatial_threshold = spatial_threshold_px
        self.deflection_threshold = deflection_threshold_deg

    def _is_coherent_track(self, points: List[List[float]], max_step_px: float = 80.0) -> bool:
        """Verifies that consecutive tracked coordinates form a physically plausible ball trajectory."""
        if len(points) < 2:
            return True
        for i in range(1, len(points)):
            p1 = np.array(points[i - 1])
            p2 = np.array(points[i])
            if np.linalg.norm(p2 - p1) > max_step_px:
                return False
        return True

    def estimate_contact(
        self,
        pose_impact_frame: int,
        bat_detections: Dict[int, Dict[str, Any]],
        ball_detections: Dict[int, Dict[str, Any]],
        total_frames: int
    ) -> Dict[str, Any]:
        """
        Estimates visual bat-ball contact independently from pose.
        Returns single authoritative contact state, confidence, and component signals.
        """
        # Define candidate search window: +/- 15 frames around pose impact estimate (configurable)
        w_start = max(1, pose_impact_frame - 15)
        w_end = min(total_frames, pose_impact_frame + 15)
        candidate_window = list(range(w_start, w_end + 1))

        # Check raw detections in window
        raw_ball_frames = sorted([f for f in candidate_window if f in ball_detections and ball_detections[f] is not None])
        raw_bat_frames = sorted([f for f in candidate_window if f in bat_detections and bat_detections[f] is not None])

        # Short-gap trajectory interpolation for ball if gap <= 5 frames between pre- and post-contact tracks
        effective_ball_detections = dict(ball_detections)
        if len(raw_ball_frames) >= 2:
            for i in range(len(raw_ball_frames) - 1):
                f1 = raw_ball_frames[i]
                f2 = raw_ball_frames[i + 1]
                gap = f2 - f1
                p1 = np.array(ball_detections[f1]['center'], dtype=float)
                p2 = np.array(ball_detections[f2]['center'], dtype=float)
                step_dist = float(np.linalg.norm(p2 - p1))
                if 1 < gap <= 6 and (step_dist / gap) <= 80.0:
                    r1 = float(ball_detections[f1].get('radius', 5.0))
                    r2 = float(ball_detections[f2].get('radius', 5.0))
                    for step, gf in enumerate(range(f1 + 1, f2), 1):
                        alpha = step / float(gap)
                        interp_c = (1.0 - alpha) * p1 + alpha * p2
                        interp_r = (1.0 - alpha) * r1 + alpha * r2
                        effective_ball_detections[gf] = {
                            'center': [int(interp_c[0]), int(interp_c[1])],
                            'radius': interp_r,
                            'confidence': 0.60,
                            'is_interpolated': True
                        }

        # Update candidate ball frames with interpolated positions
        interp_ball_frames = sorted([f for f in candidate_window if f in effective_ball_detections and effective_ball_detections[f] is not None])
        ball_pts = [effective_ball_detections[f]['center'] for f in interp_ball_frames]
        ball_track_coherent = self._is_coherent_track(ball_pts) if len(ball_pts) >= 2 else False

        # Ball tracking is AVAILABLE near contact if there are at least 3 detections with continuous track
        # OR at least 2 frames near bat with coherent track
        ball_available = (len(interp_ball_frames) >= 3 and ball_track_coherent) or (
            len(interp_ball_frames) >= 2 and ball_track_coherent and any(
                f in bat_detections and bat_detections[f] is not None for f in interp_ball_frames
            )
        )
        bat_available = len(raw_bat_frames) >= 1

        # Signals dictionary conforming to Step 14
        signals = {
            'spatial_proximity': None,
            'temporal_alignment': None,
            'relative_velocity': None,
            'ball_deflection': None,
            'bat_consistency': None
        }

        # If ball or bat tracking is not available near contact, status MUST be UNKNOWN
        if not ball_available or not bat_available:
            bat_consistency_val = None
            if bat_available:
                bat_confs = [bat_detections[f].get('confidence', 0.8) for f in raw_bat_frames if 'confidence' in bat_detections[f]]
                bat_consistency_val = round(float(np.mean(bat_confs)), 2) if bat_confs else 0.80
                signals['bat_consistency'] = bat_consistency_val

            valid_signals_count = sum(1 for v in signals.values() if v is not None)
            evidence_cov = round(valid_signals_count / 5.0, 2)

            if not ball_available and not bat_available:
                reason = "Neither ball nor bat tracking available around candidate contact window."
            elif not ball_available:
                reason = "Ball tracking unavailable around the biomechanical impact window (insufficient continuous trajectory evidence)."
            else:
                reason = "Bat tracking unavailable around candidate contact window."

            return {
                'status': STATUS_UNKNOWN,
                'confidence': 41.0,
                'confidence_score': 41.0,
                'estimated_frame': None,
                'visual_contact_frame': None,
                'pose_impact_frame': pose_impact_frame,
                'frame_difference': None,
                'ball_tracking_available': False,
                'bat_tracking_available': bat_available,
                'evidence_coverage': evidence_cov,
                'signals': signals,
                'reason': reason
            }

        # Analyze frame-by-frame candidate proximity within window
        frame_scores = []
        for f in candidate_window:
            if f in effective_ball_detections and f in bat_detections:
                ball = effective_ball_detections[f]
                bat = bat_detections[f]
                if ball and bat and 'segment' in bat:
                    b_pt = np.array(ball['center'])
                    seg = bat['segment']
                    dist = point_to_segment_distance(b_pt, seg[0], seg[1])

                    # Spatial proximity (0.0 to 1.0)
                    s_prox = float(np.clip(1.0 - (dist / max(1.0, self.spatial_threshold)), 0.0, 1.0))

                    # Temporal alignment (0.0 to 1.0)
                    t_diff = abs(f - pose_impact_frame)
                    t_score = max(0.0, 1.0 - (t_diff * 0.10))

                    frame_scores.append({
                        'frame_idx': f,
                        'distance_px': round(dist, 2),
                        'spatial_score': s_prox,
                        'temporal_score': t_score,
                        'combined_raw': s_prox * 0.65 + t_score * 0.35
                    })

        if not frame_scores:
            bat_confs = [bat_detections[f].get('confidence', 0.8) for f in raw_bat_frames if 'confidence' in bat_detections[f]]
            signals['bat_consistency'] = round(float(np.mean(bat_confs)), 2) if bat_confs else 0.80
            valid_signals_count = sum(1 for v in signals.values() if v is not None)
            return {
                'status': STATUS_UNKNOWN,
                'confidence': 41.0,
                'confidence_score': 41.0,
                'estimated_frame': None,
                'visual_contact_frame': None,
                'pose_impact_frame': pose_impact_frame,
                'frame_difference': None,
                'ball_tracking_available': ball_available,
                'bat_tracking_available': bat_available,
                'evidence_coverage': round(valid_signals_count / 5.0, 2),
                'signals': signals,
                'reason': 'Bat and ball could not be jointly localized within contact window.'
            }

        # Sort candidate frames primarily by spatial proximity (distance), then temporal proximity
        frame_scores.sort(key=lambda x: (x['spatial_score'] * 0.70 + x['temporal_score'] * 0.30), reverse=True)
        best_cand = frame_scores[0]
        best_f = best_cand['frame_idx']
        min_dist = best_cand['distance_px']

        # 1. Spatial proximity signal
        if min_dist <= self.spatial_threshold:
            s_prox = float(np.clip(1.0 - (min_dist / max(1.0, self.spatial_threshold)), 0.0, 1.0))
            signals['spatial_proximity'] = round(s_prox, 2)
        elif min_dist <= self.spatial_threshold * 1.8:
            signals['spatial_proximity'] = 0.0
        else:
            signals['spatial_proximity'] = 0.0

        # 2. Temporal alignment signal
        t_diff = abs(best_f - pose_impact_frame)
        if min_dist > self.spatial_threshold * 1.8:
            signals['temporal_alignment'] = None
        else:
            t_score = max(0.0, 1.0 - (t_diff * 0.10))
            signals['temporal_alignment'] = round(t_score, 2)

        # 3. Trajectory change & Relative velocity analysis around best_f (wider window +/- 10 frames)
        incoming_pts = [effective_ball_detections[f]['center'] for f in range(max(1, best_f - 10), best_f) if f in effective_ball_detections and effective_ball_detections[f]]
        outgoing_pts = [effective_ball_detections[f]['center'] for f in range(best_f + 1, min(total_frames, best_f + 11)) if f in effective_ball_detections and effective_ball_detections[f]]

        has_incoming = len(incoming_pts) >= 2 and self._is_coherent_track(incoming_pts)
        has_outgoing = len(outgoing_pts) >= 2 and self._is_coherent_track(outgoing_pts)

        deflection_deg = 0.0
        s_traj = None
        v_score = None

        if has_incoming and has_outgoing:
            deflection_deg = compute_trajectory_deflection(incoming_pts, outgoing_pts)
            if min_dist <= self.spatial_threshold * 1.5:
                if 12.0 <= deflection_deg <= 135.0:
                    s_traj = float(np.clip(deflection_deg / self.deflection_threshold, 0.0, 1.0))
                else:
                    s_traj = 0.20
                signals['ball_deflection'] = round(s_traj, 2)

                v_in = np.array(incoming_pts[-1]) - np.array(incoming_pts[0])
                v_out = np.array(outgoing_pts[-1]) - np.array(outgoing_pts[0])
                dv = float(np.linalg.norm(v_out - v_in))
                v_score = float(np.clip(dv / 25.0, 0.0, 1.0))
                signals['relative_velocity'] = round(v_score, 2)
            else:
                # Trajectory deflection far from the bat cannot be contact evidence
                signals['ball_deflection'] = None
                signals['relative_velocity'] = None
        else:
            signals['ball_deflection'] = None
            signals['relative_velocity'] = None

        # 5. Bat consistency signal
        if best_f in bat_detections and bat_detections[best_f] is not None:
            bat_conf = float(bat_detections[best_f].get('confidence', 0.85))
            signals['bat_consistency'] = round(bat_conf, 2)
        else:
            signals['bat_consistency'] = None

        # Calculate evidence coverage
        valid_signals_count = sum(1 for v in signals.values() if v is not None)
        evidence_cov = round(valid_signals_count / 5.0, 2)

        # Decide authoritative status based on physical evidence
        if min_dist <= self.spatial_threshold:
            # Candidate ball position meets spatial contact criterion
            contact_indicators = []
            if signals['spatial_proximity'] is not None:
                contact_indicators.append(signals['spatial_proximity'] * 0.40)
            if signals['temporal_alignment'] is not None:
                contact_indicators.append(signals['temporal_alignment'] * 0.20)
            if signals['ball_deflection'] is not None:
                contact_indicators.append(signals['ball_deflection'] * 0.25)
            if signals['bat_consistency'] is not None:
                contact_indicators.append(signals['bat_consistency'] * 0.15)

            calc_conf = round(float(np.clip(sum(contact_indicators) * 100.0, 50.0, 95.0)), 1)

            if s_traj is not None and s_traj >= 0.50:
                calc_conf = max(calc_conf, 88.0)

            if calc_conf >= 60.0 or (s_traj is not None and s_traj >= 0.60):
                status = STATUS_CONTACT
                final_confidence = calc_conf
                reason = f"Contact confirmed at frame {best_f} (distance {min_dist}px, deflection {round(deflection_deg, 1)}°)."
                est_frame = best_f
                diff_frame = best_f - pose_impact_frame
            else:
                status = STATUS_UNKNOWN
                final_confidence = 50.0
                reason = f"Ball near bat ({min_dist}px at frame {best_f}), but visual trajectory evidence is inconclusive."
                est_frame = None
                diff_frame = None

        elif min_dist > self.spatial_threshold * 1.5:
            # Ball definitely missed the bat — only claim NO_CONTACT if tracking reliably covered the pass
            if (has_incoming and has_outgoing) or (len(raw_ball_frames) >= 4 and ball_track_coherent):
                status = STATUS_NO_CONTACT
                final_confidence = 92.0
                reason = "Ball remained outside the validated bat-contact region throughout the contact window."
            else:
                # Sparse/broken tracking cannot prove a miss
                status = STATUS_UNKNOWN
                final_confidence = 41.0
                reason = "Insufficient continuous ball trajectory around candidate contact window."
            est_frame = None
            diff_frame = None

        else:
            # Borderline distance (between 1.0x and 1.5x threshold)
            status = STATUS_UNKNOWN
            final_confidence = 45.0
            reason = f"Borderline contact proximity ({min_dist}px). Visual evidence is inconclusive."
            est_frame = None
            diff_frame = None

        return {
            'status': status,
            'confidence': final_confidence,
            'confidence_score': final_confidence,
            'estimated_frame': est_frame,
            'visual_contact_frame': est_frame,
            'pose_impact_frame': pose_impact_frame,
            'frame_difference': diff_frame,
            'ball_tracking_available': ball_available,
            'bat_tracking_available': bat_available,
            'evidence_coverage': evidence_cov,
            'signals': {
                'frame': best_f,
                'bat_ball_distance_px': min_dist,
                'spatial_proximity': signals['spatial_proximity'],
                'temporal_alignment': signals['temporal_alignment'],
                'relative_velocity': signals['relative_velocity'],
                'ball_deflection': signals['ball_deflection'],
                'bat_consistency': signals['bat_consistency'],
                'normalized_spatial_score': signals['spatial_proximity'],
                'temporal_alignment_score': signals['temporal_alignment'],
                'relative_velocity_score': signals['relative_velocity'],
                'ball_trajectory_change_score': signals['ball_deflection'],
                'bat_trajectory_consistency_score': signals['bat_consistency'],
                'final_contact_score': final_confidence
            },
            'reason': reason
        }
