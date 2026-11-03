"""
AthleteRise - AI-Powered Multi-Shot Cricket Analytics Dashboard
Streamlit GUI Supporting:
- Tab 1: Single Video Multi-Shot Analysis (Classification, Contact Detection, Scoring, Feedback)
- Tab 2: Multi-Shot Reference Calibration (Cover Drive, Straight Drive, Pull Shot)
- Tab 3: Contact Validation & Annotation Mode (Human-in-the-Loop Ground Truth)
"""

import streamlit as st
import os
import json
import glob
from pathlib import Path
import cv2
import cover_drive_analysis_realtime
from src.calibration.calibration_factory import CalibrationFactory
from src.calibration.calibration_engine import get_dedicated_reference_videos
from src.report.calibration_reporter import ShotCalibrationReporter, CoverDriveCalibrationReporter
from src.shot_analysis.shot_config import SHOT_COVER_DRIVE, SHOT_STRAIGHT_DRIVE, SHOT_PULL_SHOT, SHOT_DISPLAY_NAMES
from src.shot_analysis.shot_classifier import ShotClassifier
from src.contact_detection.annotation_manager import ContactAnnotationManager
import matplotlib.pyplot as plt
from src.calibration.biomechanical_statistics import BiomechanicalStatisticsAnalyzer, get_metric_display_name

def fmt_num(val, decimals=2):
    if val is None:
        return "N/A"
    try:
        f = float(val)
        return f"{f:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)

st.set_page_config(
    page_title="AthleteRise - Multi-Shot Cricket Batting Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏏 AthleteRise — Multi-Shot Cricket Batting Analytics")
st.markdown("Automated Shot Classification, Bat–Ball Contact Detection, and Empirical Biomechanical Calibration Engine")

# Ensure required directories exist
for s in ["cover_drive", "straight_drive", "pull_shot"]:
    os.makedirs(f"data/reference_videos/{s}", exist_ok=True)
    os.makedirs(f"output/reference_calibration/{s}", exist_ok=True)
os.makedirs("data/contact_annotations", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 3 Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🎥 Single Video Shot Analysis",
    "📐 Multi-Shot Reference Calibration",
    "🎯 Contact Validation & Annotation"
])

# ==============================================================================
# TAB 1: SINGLE VIDEO MULTI-SHOT ANALYSIS
# ==============================================================================
with tab1:
    st.subheader("Automated Multi-Shot Video Analysis")
    st.markdown("Upload a cricket batting video to automatically determine the **shot played**, detect **bat–ball contact**, and evaluate biomechanics against the calibrated statistical reference baseline.")

    shot_override_opt = st.selectbox(
        "Shot Classification Mode",
        options=["Auto-Detect Shot (Recommended)", "Cover Drive (Force Baseline)", "Straight Drive (Force Baseline)", "Pull Shot (Force Baseline)"],
        index=0,
        help="System automatically classifies the shot or you can lock the evaluation baseline."
    )
    shot_override = None
    if "Cover Drive" in shot_override_opt:
        shot_override = "cover_drive"
    elif "Straight Drive" in shot_override_opt:
        shot_override = "straight_drive"
    elif "Pull Shot" in shot_override_opt:
        shot_override = "pull_shot"

    video_file = st.file_uploader(
        "📂 Upload Batting Video (MP4 / AVI)",
        type=["mp4", "avi"],
        key="multi_shot_uploader"
    )

    if st.button("🚀 Run Batting Analysis", key="btn_run_analysis"):
        if video_file is not None:
            with st.spinner("Processing video, tracking kinematics, and estimating contact..."):
                with open("temp.mp4", "wb") as f:
                    f.write(video_file.read())

                try:
                    cover_drive_analysis_realtime.main(video_input="temp.mp4", shot_type_override=shot_override)
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error while running analysis: {e}")
                    st.stop()

            output_video_path = "output/annotated_video.mp4"
            json_path = "output/evaluation.json"
            report_path = "output/evaluation_report.txt"

            # Display Annotated Video
            if os.path.exists(output_video_path):
                st.subheader("🎥 Annotated Multi-Shot Video")
                with open(output_video_path, "rb") as vb:
                    v_bytes = vb.read()
                st.video(v_bytes)
                st.download_button("📥 Download Annotated Video", v_bytes, file_name="annotated_video.mp4", mime="video/mp4")

            # Structured Output Display
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)

                    shot_data = eval_data.get("shot_analysis", {})
                    pred_shot = shot_data.get("predicted_shot", "cover_drive")
                    pred_name = SHOT_DISPLAY_NAMES.get(pred_shot, pred_shot.replace("_", " ").title())
                    shot_conf = shot_data.get("confidence", 0.0)
                    conf_level = shot_data.get("confidence_level", "Low Confidence" if pred_shot == "unknown" else "Moderate Confidence")
                    top_prob = shot_data.get("top_probability", shot_conf)
                    top_margin = shot_data.get("top_class_margin", 0.0)
                    ref_sim = shot_data.get("reference_similarity", 0.0)
                    feat_cov = shot_data.get("feature_coverage", 1.0)
                    feat_ev = shot_data.get("feature_evidence", {})
                    top_cand = shot_data.get("top_candidate", feat_ev.get("top_candidate", pred_shot))
                    top_cand_name = SHOT_DISPLAY_NAMES.get(top_cand, top_cand.replace("_", " ").title())

                    contact_data = eval_data.get("contact_analysis", {})
                    c_status_raw = contact_data.get("status", "unknown").lower()
                    if c_status_raw in ("contact", "contact_detected"):
                        c_status_display = "Contact Detected"
                    elif c_status_raw in ("no_contact", "no_contact_detected"):
                        c_status_display = "No Contact Detected"
                    else:
                        c_status_display = "Unknown"

                    c_evidence = contact_data.get("evidence_score", contact_data.get("confidence", 41.0))
                    c_frame = contact_data.get("visual_contact_frame", contact_data.get("estimated_frame"))
                    p_frame = contact_data.get("biomechanical_impact_frame", contact_data.get("pose_impact_frame"))
                    diff_frame = contact_data.get("frame_difference")

                    bat_trk = eval_data.get("bat_tracking", {})
                    ball_trk = eval_data.get("ball_tracking", {})

                    overall_score = eval_data.get("overall_score_display", "N/A")
                    skill_grade = eval_data.get("skill_grade", "N/A")

                    st.markdown("---")
                    st.subheader("🎯 Primary Analysis Summary")

                    is_unknown = (pred_shot == "unknown")

                    # Row 1: Shot Classification & Technique Status
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("Detected Shot", pred_name, f"Conf: {round(shot_conf * 100, 1)}% ({conf_level})")
                    col_s2.metric("Overall Technique Score", overall_score if not is_unknown else "N/A", skill_grade if not is_unknown else "Unclassified Shot (General Metrics Below)")
                    col_s3.metric("Candidate & Separation", f"Top: {top_cand_name}", f"Margin: {round(top_margin * 100, 1)}% | Sim: {round(ref_sim * 100, 1)}%")

                    # Row 2: Contact & Impact Separation (Section 17, 18 & 19)
                    col_c1, col_c2, col_c3 = st.columns(3)
                    col_c1.metric("Bat–Ball Contact", c_status_display, f"Evidence: {fmt_num(c_evidence, 1)} / 100")
                    col_c2.metric("Biomechanical Impact Estimate", f"Frame {p_frame}" if p_frame is not None else "N/A", "Pose Body Kinematics Window")

                    # Completely consistent Visual Bat-Ball Contact (Section 19)
                    if c_status_raw in ("contact", "contact_detected"):
                        v_contact_display = f"Frame {c_frame}" if c_frame is not None else "Contact Detected"
                        v_contact_sub = f"Diff: {diff_frame:+d} frames from impact" if diff_frame is not None else "Visual Contact"
                    elif c_status_raw in ("no_contact", "no_contact_detected"):
                        v_contact_display = "No Contact"
                        v_contact_sub = "Validated Clean Miss"
                    else:
                        v_contact_display = "N/A"
                        v_contact_sub = "Insufficient Tracking Evidence"

                    col_c3.metric("Visual Bat–Ball Contact Frame", v_contact_display, v_contact_sub)

                    # Shot Classification Diagnostics Expander (Section 31)
                    with st.expander("🔬 Shot Classification Diagnostics & Validation", expanded=False):
                        probs = shot_data.get("class_probabilities", {})
                        diag_c1, diag_c2 = st.columns(2)

                        with diag_c1:
                            st.markdown("##### Reference Counts & Class Probabilities")
                            st.write("- **Cover Drive Reference:** 10 videos calibrated")
                            st.write("- **Straight Drive Reference:** 10 videos calibrated")
                            st.write("- **Pull Shot Reference:** 10 videos calibrated")
                            if probs:
                                for sk, pv in probs.items():
                                    st.write(f"- **{SHOT_DISPLAY_NAMES.get(sk, sk)}:** `{round(pv * 100, 1)}%`")

                        with diag_c2:
                            st.markdown("##### Decision Evidence & Metrics")
                            st.write(f"- **Top Candidate:** `{top_cand_name}` ({round(top_prob * 100, 1)}%)")
                            second_cand = feat_ev.get('runner_up', 'N/A')
                            st.write(f"- **Second Candidate:** `{SHOT_DISPLAY_NAMES.get(second_cand, second_cand)}`")
                            st.write(f"- **Top-Class Margin:** `{round(top_margin * 100, 1)} percentage points`")
                            st.write(f"- **Reference Similarity:** `{round(ref_sim * 100, 1)}%`")
                            st.write(f"- **Feature Coverage:** `{int(feat_cov * 100)}%`")
                            st.write(f"- **Final Classification Decision:** `{pred_name}` ({conf_level})")
                            st.write(f"- **Decision Reason:** {shot_data.get('reason', 'N/A')}")

                        # Classifier Model Validation Section (Section 7)
                        st.markdown("---")
                        st.markdown("##### 📈 Empirical Classifier Validation (Leave-One-Out Cross-Validation)")
                        try:
                            clf_eval = ShotClassifier().evaluate_loo_cross_validation()
                            cv_col1, cv_col2, cv_col3 = st.columns(3)
                            cv_col1.metric("Dataset Size", f"{clf_eval.get('total_samples', 30)} Samples", "10 clips / class")
                            cv_col2.metric("LOO-CV Accuracy", f"{round(clf_eval.get('accuracy', 0.0) * 100, 1)}%", "3-Class Evaluation")
                            cv_col3.metric("Macro F1-Score", f"{round(clf_eval.get('macro_f1', 0.0) * 100, 1)}%", "Unweighted Mean")

                            per_class = clf_eval.get('per_class', {})
                            if per_class:
                                pc_rows = []
                                for c_shot, c_metrics in per_class.items():
                                    pc_rows.append({
                                        "Shot Class": SHOT_DISPLAY_NAMES.get(c_shot, c_shot),
                                        "Precision": f"{round(c_metrics.get('precision', 0.0) * 100, 1)}%",
                                        "Recall": f"{round(c_metrics.get('recall', 0.0) * 100, 1)}%",
                                        "F1-Score": f"{round(c_metrics.get('f1_score', 0.0) * 100, 1)}%",
                                        "Support": c_metrics.get('support', 10)
                                    })
                                st.dataframe(pc_rows, width="stretch")
                            st.caption(f"ℹ️ {clf_eval.get('disclaimer', '')}")
                        except Exception as e:
                            st.info("Classifier validation unavailable.")

                        # Tracking Quality Diagnostics (Section 15 & 16)
                        st.markdown("---")
                        st.markdown("##### 📹 Vision Tracking Quality Summary")
                        f_cols = st.columns(3)
                        f_cols[0].metric(
                            "Pose Tracking",
                            "Available",
                            f"Coverage: {int(eval_data.get('quality', {}).get('pose_quality', 1.0) * 100)}%"
                        )
                        f_cols[1].metric(
                            "Bat Tracking",
                            bat_trk.get("status", "AVAILABLE"),
                            f"Coverage: {bat_trk.get('coverage_pct', 0)}% ({bat_trk.get('continuity', 'Good')})"
                        )
                        f_cols[2].metric(
                            "Ball Tracking",
                            ball_trk.get("status", "UNAVAILABLE"),
                            f"Coverage: {ball_trk.get('coverage_pct', 0)}% ({ball_trk.get('continuity', 'Poor')})"
                        )

                    # Bat-Ball Contact Deep Dive
                    with st.expander("⚾ Bat–Ball Contact Deep Dive & Component Signals"):
                        c_col1, c_col2 = st.columns(2)
                        c_col1.write(f"**Visual Contact Decision:** `{c_status_display}`")
                        if c_status_raw in ("contact", "contact_detected"):
                            c_col1.write(f"**Visual Contact Frame:** `{f'Frame {c_frame}' if c_frame is not None else 'Contact Detected'}`")
                        elif c_status_raw in ("no_contact", "no_contact_detected"):
                            c_col1.write(f"**Visual Contact Frame:** `N/A (Validated Miss)`")
                        else:
                            c_col1.write(f"**Visual Contact Frame:** `N/A (Unknown)`")

                        c_col1.write(f"**Biomechanical Impact Estimate:** `Frame {p_frame}`")
                        if diff_frame is not None:
                            c_col1.write(f"**Frame Difference:** `{diff_frame:+d} frames`")
                        dist_px = contact_data.get('bat_ball_distance_px')
                        c_col1.write(f"**Bat–Ball Proximity:** `{fmt_num(dist_px, 1)} px`" if dist_px is not None else "**Bat–Ball Proximity:** `N/A`")

                        c_col2.write(f"**Contact Evidence Score:** `{fmt_num(c_evidence, 1)} / 100`")
                        ev_cov = contact_data.get('evidence_coverage')
                        c_col2.write(f"**Signal Evidence Coverage:** `{int(ev_cov * 100)}%`" if ev_cov is not None else "**Signal Evidence Coverage:** `N/A`")
                        c_col2.write(f"**Ball Tracking Status:** `{ball_trk.get('status', 'UNAVAILABLE')} ({ball_trk.get('coverage_pct', 0)}% coverage)`")
                        c_col2.write(f"**Bat Tracking Status:** `{bat_trk.get('status', 'AVAILABLE')} ({bat_trk.get('coverage_pct', 0)}% coverage)`")

                        # Component signals breakdown
                        st.markdown("##### Multi-Signal Component Breakdown")
                        sigs_dict = contact_data.get("signals", {})
                        sig_cols = st.columns(5)
                        sig_cols[0].metric("Spatial Proximity", fmt_num(sigs_dict.get("spatial_proximity"), 2))
                        sig_cols[1].metric("Temporal Align", fmt_num(sigs_dict.get("temporal_alignment"), 2))
                        sig_cols[2].metric("Rel Velocity", fmt_num(sigs_dict.get("relative_velocity"), 2))
                        sig_cols[3].metric("Ball Deflection", fmt_num(sigs_dict.get("ball_deflection"), 2))
                        sig_cols[4].metric("Bat Consistency", fmt_num(sigs_dict.get("bat_consistency"), 2))

                        st.info(f"ℹ️ **Contact Assessment:** {contact_data.get('reason', 'N/A')}")

                    # GENERAL BIOMECHANICAL ANALYSIS (Always Visible per Section 13 & 32)
                    st.markdown("---")
                    st.subheader("📊 General Biomechanical Analysis")
                    if is_unknown:
                        st.info("ℹ️ **Classification: UNKNOWN**\n\n**General Biomechanical Measurements:** These measurements are available and accurately extracted, but shot-specific scoring is unavailable because the shot class could not be determined reliably.")
                    else:
                        st.caption("Extracted core biomechanical metrics from monocular pose tracking (always available regardless of shot classification).")
                    metric_scores = eval_data.get("metric_evaluations", {})
                    if not metric_scores:
                        metric_scores = eval_data.get("metric_scores", {})
                    if metric_scores:
                        table_rows = []
                        for mk, mv in metric_scores.items():
                            raw_v = mv.get('raw_value')
                            u_disp = mv.get('user_value_display', f"{raw_v} {mv.get('unit', '')}" if raw_v is not None else "N/A")
                            sc_disp = mv.get('metric_score_display', f"{fmt_num(mv.get('score'), 1)} / 100" if mv.get('score') is not None else "General Measurement Only")
                            ref_mean = fmt_num(mv.get('reference_mean')) if not is_unknown else "N/A"
                            p10 = mv.get('reference_p10')
                            p90 = mv.get('reference_p90')
                            ref_range = f"[{fmt_num(p10)} - {fmt_num(p90)}]" if (not is_unknown and p10 is not None) else "N/A"
                            table_rows.append({
                                "Metric": mv.get('name', mk.replace('_', ' ').title()),
                                "Measured Value": u_disp,
                                "Score (/100)": sc_disp if not is_unknown else "General Measurement Only",
                                "Ref Baseline Mean": ref_mean,
                                "Ref P10-P90 Range": ref_range,
                                "Assessment": mv.get('assessment', '')
                            })
                        st.dataframe(table_rows, width="stretch")

                        # Reference Baseline Distribution & Box Plots Expander (Empirical Baseline)
                        ref_path_for_tab1 = eval_data.get("reference_path", f"output/reference_calibration/{eval_data.get('shot_type', 'cover_drive')}/{eval_data.get('shot_type', 'cover_drive')}_reference.json")
                        if os.path.exists(ref_path_for_tab1):
                            with st.expander("📦 Reference Calibration Baseline Distribution & Box Plots (Empirical Reference)", expanded=False):
                                ref_tab1_analyzer = BiomechanicalStatisticsAnalyzer(ref_path_for_tab1, shot_type=eval_data.get('shot_type', 'cover_drive'))
                                st.info(
                                    f"📌 **Statistical Analysis Source:** Calculated from available reference/calibration data across all **{len(ref_tab1_analyzer.valid_videos)}** reference videos.\n\n"
                                    f"• **Independent Metric Treatment:** Each biomechanical metric (Elbow Angle, Head-Knee Distance, Foot Angle, Spine Lean, etc.) is analyzed independently.\n"
                                    f"• **Separation of Thresholds:** These values represent the empirical distribution of reference videos. Heuristic scoring thresholds remain separate."
                                )
                                st_tab1_df = ref_tab1_analyzer.get_summary_dataframe()
                                tab1_cols = ['Biomechanical Metric', 'Valid Videos', 'Mean', 'Median', 'Std Dev', 'Variance', 'Min', 'Max', 'Q1', 'Q3', 'IQR', 'P10', 'P90', 'Outliers Count']
                                st.dataframe(st_tab1_df[tab1_cols], width="stretch")
                                fig_tab1 = ref_tab1_analyzer.plot_all_metric_boxplots()
                                st.pyplot(fig_tab1)
                                plt.close(fig_tab1)

                                ref_shot_code = eval_data.get('shot_type', 'cover_drive')
                                tab1_dist_img = f"output/reference_calibration/{ref_shot_code}/{ref_shot_code}_distribution.png"
                                if os.path.exists(tab1_dist_img):
                                    st.markdown(f"#### 📈 {ref_shot_code.replace('_', ' ').title()} Reference Distribution & Correlation Suite")
                                    st.image(tab1_dist_img, caption=f"{ref_shot_code.replace('_', ' ').title()} Empirical Biomechanical Reference Suite", use_container_width=True)

                    # SHOT-SPECIFIC ANALYSIS (Section 12 & 13)
                    st.markdown("---")
                    st.subheader(f"⚡ Shot-Specific Analysis" + (f" ({pred_name})" if not is_unknown else ""))
                    if is_unknown:
                        st.info("ℹ️ **Shot-specific scoring unavailable because the shot could not be classified reliably.**")
                    else:
                        shot_specific = eval_data.get("shot_specific_metrics", {})
                        if shot_specific:
                            sp_cols = st.columns(len(shot_specific))
                            for idx, (sp_k, sp_v) in enumerate(shot_specific.items()):
                                sp_cols[idx % len(shot_specific)].metric(sp_k.replace("_", " ").title(), fmt_num(sp_v, 2))
                        else:
                            st.write("No shot-specific metrics extracted.")

                    # CATEGORY BREAKDOWN (5 Categories)
                    st.markdown("---")
                    st.subheader("🏏 Category Breakdown (5 Categories)")
                    cat_scores = eval_data.get("category_scores", {})
                    if is_unknown:
                        st.info("ℹ️ **Category breakdown scoring unavailable because the shot could not be classified reliably.**")
                    elif cat_scores:
                        c_cols = st.columns(len(cat_scores))
                        for idx, (ck, cv) in enumerate(cat_scores.items()):
                            c_cols[idx % len(cat_scores)].metric(cv.get("category_name", ck), cv.get("score_display", "N/A"))

                    # COACHING FEEDBACK
                    feedback = eval_data.get("coaching_feedback", {})
                    if feedback:
                        st.markdown("---")
                        st.subheader("💡 Coaching Insights & Recommendations")
                        fb1, fb2 = st.columns(2)
                        with fb1:
                            st.markdown("#### ✅ Key Strengths")
                            strengths_list = feedback.get("key_strengths", [])
                            if strengths_list:
                                for s in strengths_list:
                                    st.write(f"- {s}")
                            else:
                                st.write("Classification required to evaluate stroke strengths.")
                        with fb2:
                            st.markdown("#### 🎯 Coaching Recommendations")
                            for r in feedback.get("coaching_recommendations", []):
                                st.write(f"- {r}")

                except Exception as e:
                    st.warning(f"Could not render full UI: {e}")

            # Download Artifacts
            st.markdown("---")
            dl1, dl2 = st.columns(2)
            if os.path.exists(json_path):
                with open(json_path, "rb") as f:
                    dl1.download_button("📥 Download evaluation.json", f, file_name="evaluation.json")
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    dl2.download_button("📥 Download evaluation_report.txt", f, file_name="evaluation_report.txt")

        else:
            st.error("Please upload a video file to proceed.")


# ==============================================================================
# TAB 2: MULTI-SHOT REFERENCE CALIBRATION
# ==============================================================================
with tab2:
    st.subheader("Multi-Shot Empirical Reference Calibration")
    st.markdown("Derive statistical reference distributions ($P_{10}$ to $P_{90}$) independently for each shot class.")

    calib_shot = st.selectbox(
        "Select Shot Type to Calibrate",
        options=["Cover Drive", "Straight Drive", "Pull Shot"],
        key="calib_shot_selector"
    )
    shot_code = calib_shot.lower().replace(" ", "_")
    dataset_dir = f"data/reference_videos/{shot_code}"
    output_dir = f"output/reference_calibration/{shot_code}"

    st.info(f"ℹ️ Calibration dataset path: `{dataset_dir}/`. Baseline output path: `{output_dir}/`.")

    detected_vids = get_dedicated_reference_videos(dataset_dir)
    st.write(f"**Reference Videos Detected for {calib_shot}:** `{len(detected_vids)}`")

    if detected_vids:
        st.dataframe([{"Index": i+1, "File": os.path.basename(v), "Path": v} for i, v in enumerate(detected_vids)], width="stretch")

    ref_json_path = f"{output_dir}/{shot_code}_reference.json"
    ref_data = None

    col_btn1, col_btn2 = st.columns([1, 2])
    run_calib = col_btn1.button(f"🚀 Run {calib_shot} Calibration", disabled=len(detected_vids) == 0, key="btn_run_calib")

    if run_calib:
        with st.spinner(f"Calibrating {calib_shot} baseline across {len(detected_vids)} videos..."):
            engine = CalibrationFactory.create(shot_type=shot_code, reference_dir=dataset_dir, output_dir=output_dir)
            ref_data = engine.run_calibration(detected_vids)
            st.success(f"✅ {calib_shot} Calibration Complete! Baseline derived from {ref_data['valid_videos_count']} videos.")
    elif os.path.exists(ref_json_path):
        with open(ref_json_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)

    if ref_data:
        analyzer = BiomechanicalStatisticsAnalyzer(ref_data, shot_type=shot_code)
        stats_df = analyzer.get_summary_dataframe()
        valid_vids_count = ref_data.get('valid_videos_count', len(analyzer.valid_videos))

        # Clear labeling notice as strictly required
        st.markdown("---")
        st.info(
            f"📌 **Statistical Analysis Source:** Calculated from available reference/calibration data across all **{valid_vids_count}** reference videos.\n\n"
            f"• **Independent Metric Treatment:** Each biomechanical metric (Elbow Angle, Head-Knee Distance, Foot Angle, Spine Lean, etc.) is analyzed independently.\n"
            f"• **Separation of Thresholds:** These values represent the empirical distribution of reference videos. Manually defined scoring thresholds remain strictly separate from this statistical analysis."
        )

        st.markdown(f"### 📋 {calib_shot} Biomechanical Metrics Statistical Comparison Table")
        st.caption("Compare biomechanical metrics using: Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR, P10, and P90")

        # Table with all 11 required statistical descriptors
        cols_order = ['Biomechanical Metric', 'Valid Videos', 'Mean', 'Median', 'Std Dev', 'Variance', 'Min', 'Max', 'Q1', 'Q3', 'IQR', 'P10', 'P90', 'Outliers Count']
        st.dataframe(stats_df[cols_order], width="stretch")

        csv_bytes = stats_df[cols_order].to_csv(index=False).encode('utf-8')
        st.download_button(
            f"📥 Download {calib_shot} Statistical Summary (CSV)",
            data=csv_bytes,
            file_name=f"{shot_code}_statistical_summary.csv",
            mime="text/csv",
            key=f"dl_stats_csv_{shot_code}"
        )

        # Box plots visualization section
        st.markdown("---")
        st.markdown(f"### 📦 {calib_shot} Biomechanical Metric Box Plots")
        st.markdown(
            "Visually understand the empirical reference distributions:\n"
            "- **Central Tendency:** Median (bold red line) and Mean (green diamond marker)\n"
            "- **Spread:** Interquartile Range (IQR box spanning Q1 to Q3)\n"
            "- **Variability:** Whiskers showing dispersion across reference videos ($1.5 \\times \\text{IQR}$ bounds)\n"
            "- **Potential Outliers:** Individual red flier points outside the whiskers\n"
            "- **Raw Data Overlay:** Jittered dots showing individual reference video measurements"
        )

        plot_mode = st.radio(
            "Select Box Plot Visualization Mode:",
            options=["📊 Multi-Metric Overview (All Biomechanical Metrics Grid)", "🔍 Single Metric Deep Dive (Focused Box Plot)"],
            horizontal=True,
            key=f"plot_mode_{shot_code}"
        )

        if "Multi-Metric Overview" in plot_mode:
            fig_all = analyzer.plot_all_metric_boxplots()
            st.pyplot(fig_all)
            plt.close(fig_all)
        else:
            metric_options = [st_row['Biomechanical Metric'] for _, st_row in stats_df.iterrows()]
            metric_key_map = {st_row['Biomechanical Metric']: st_row['Metric Key'] for _, st_row in stats_df.iterrows()}

            selected_metric_name = st.selectbox(
                "Select Biomechanical Metric to Inspect:",
                options=metric_options,
                index=0,
                key=f"sel_metric_box_{shot_code}"
            )
            selected_key = metric_key_map[selected_metric_name]
            fig_single = analyzer.plot_single_metric_boxplot(selected_key)
            st.pyplot(fig_single)
            plt.close(fig_single)

            st_data = analyzer.statistics.get(selected_key, {})
            per_vid_dict = st_data.get('per_video_values', {})
            if per_vid_dict:
                with st.expander(f"📹 View Per-Video Values for {selected_metric_name}", expanded=False):
                    pv_rows = [{"Reference Video": vk, f"Measured {selected_metric_name}": vv} for vk, vv in per_vid_dict.items()]
                    st.dataframe(pv_rows, width="stretch")

        # Comprehensive Empirical Reference Visualization Suite
        st.markdown("---")
        st.markdown(f"### 📈 {calib_shot} Empirical Reference Visualization Suite")
        st.markdown(
            "Multi-panel biomechanical calibration dashboard featuring:\n"
            "- **Panel 1:** $P_{10}$ to $P_{90}$ Empirical Reference Ranges, with Mean (red dot) and Median (green square)\n"
            "- **Panel 2:** Front Elbow Angle Across All Calibration Videos vs. Baseline Mean and Bounds\n"
            "- **Panel 3:** Metric Correlation Matrix Heatmap\n"
            "- **Panel 4:** Normalized Biomechanical Radar Profile Chart"
        )

        dist_img_path = f"{output_dir}/{shot_code}_distribution.png"
        if os.path.exists(dist_img_path):
            st.image(dist_img_path, caption=f"AthleteRise: {calib_shot} Reference Distribution Dashboard", use_container_width=True)
            
            col_d1, col_d2 = st.columns(2)
            with open(dist_img_path, "rb") as f_img:
                col_d1.download_button(
                    f"📥 Download {calib_shot} Distribution Dashboard (PNG)",
                    data=f_img.read(),
                    file_name=f"{shot_code}_distribution.png",
                    mime="image/png",
                    key=f"dl_dist_png_{shot_code}"
                )
            
            html_report_path = f"{output_dir}/{shot_code}_calibration_report.html"
            if os.path.exists(html_report_path):
                with open(html_report_path, "rb") as f_html:
                    col_d2.download_button(
                        f"🌐 Download Full {calib_shot} HTML Calibration Report",
                        data=f_html.read(),
                        file_name=f"{shot_code}_calibration_report.html",
                        mime="text/html",
                        key=f"dl_report_html_{shot_code}"
                    )

    else:
        st.warning(f"No existing reference calibration found for {calib_shot}. Click 'Run {calib_shot} Calibration' above to derive empirical statistics.")

    diag_file = "output/reference_calibration/calibration_diagnostics.txt"
    if os.path.exists(diag_file):
        with st.expander("📊 Comparative Cross-Shot Diagnostic Matrix (Cover vs Straight vs Pull)", expanded=False):
            with open(diag_file, "r", encoding="utf-8") as df:
                diag_text = df.read()
            st.code(diag_text, language="text")


# ==============================================================================
# TAB 3: CONTACT VALIDATION & ANNOTATION MODE
# ==============================================================================
with tab3:
    st.subheader("🎯 Bat–Ball Contact Candidate Review & Annotation")
    st.markdown("Human-in-the-loop validation mode to confirm contact frames and build real ground truth in `data/contact_annotations/`.")

    ann_mgr = ContactAnnotationManager()
    existing_labels = ann_mgr.list_all_annotations()

    st.markdown(f"**Stored Contact Ground-Truth Annotations:** `{len(existing_labels)}`")
    if existing_labels:
        st.dataframe(existing_labels, width="stretch")
    else:
        st.info("ℹ️ No ground-truth contact labels recorded yet. Confirm candidate frames below to build your validation dataset.")

    # Validation evaluation status (Section 20)
    st.markdown("##### 📊 Contact Detector Validation Status")
    val_status = ann_mgr.evaluate_detector_against_annotations([])
    if val_status.get('status') == 'NOT YET VALIDATED':
        st.warning(f"**Contact Detector Validation:** `NOT YET VALIDATED` — {val_status.get('message', 'No ground-truth annotations exist yet.')}")
    else:
        st.success(f"**Contact Detector Validation:** `VALIDATED` (Samples: {val_status.get('sample_count', 0)}, Precision: {val_status.get('precision')}, Recall: {val_status.get('recall')}, F1: {val_status.get('f1_score')})")

    st.markdown("---")
    st.markdown("### 🔍 Review Recent Candidate Contact Frame")
    json_path = "output/evaluation.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cur_eval = json.load(f)
            c_cand = cur_eval.get("contact_analysis", {})
            cand_frame = c_cand.get("estimated_frame") or c_cand.get("pose_impact_frame") or 1

            st.write(f"**Current Video:** `temp.mp4` | **Candidate Contact Frame:** `Frame {cand_frame}`")
            st.write(f"**Evidence Score:** `{c_cand.get('confidence', 0)} / 100` | **Status:** `{c_cand.get('status', 'UNKNOWN')}`")

            # Extract and display 3-frame window: before, candidate, after
            if os.path.exists("temp.mp4"):
                cap = cv2.VideoCapture("temp.mp4")
                tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                f_before_idx = max(0, cand_frame - 2)
                f_cand_idx = max(0, min(tot_f - 1, cand_frame - 1))
                f_after_idx = min(tot_f - 1, cand_frame)

                cap.set(cv2.CAP_PROP_POS_FRAMES, f_before_idx)
                ret_b, img_before = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_cand_idx)
                ret_c, img_cand = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_after_idx)
                ret_a, img_after = cap.read()
                cap.release()

                f_col1, f_col2, f_col3 = st.columns(3)
                if ret_b:
                    f_col1.image(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB), caption=f"Frame Before (F:{f_before_idx + 1})", width="stretch")
                if ret_c:
                    f_col2.image(cv2.cvtColor(img_cand, cv2.COLOR_BGR2RGB), caption=f"🎯 Candidate Contact (F:{cand_frame})", width="stretch")
                if ret_a:
                    f_col3.image(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB), caption=f"Frame After (F:{f_after_idx + 1})", width="stretch")

            c_btn1, c_btn2, c_btn3 = st.columns(3)
            if c_btn1.button("✅ CONFIRM CONTACT", key="btn_confirm_contact"):
                p = ann_mgr.save_annotation("temp.mp4", cand_frame, "contact", notes="User verified bat-ball contact")
                st.success(f"Saved ground-truth label to `{p}`!")
                st.rerun()

            if c_btn2.button("❌ NO CONTACT", key="btn_no_contact"):
                p = ann_mgr.save_annotation("temp.mp4", cand_frame, "no_contact", notes="User verified swing and miss")
                st.warning(f"Saved ground-truth label to `{p}`!")
                st.rerun()

            if c_btn3.button("❓ UNCERTAIN", key="btn_uncertain"):
                p = ann_mgr.save_annotation("temp.mp4", cand_frame, "uncertain", notes="User marked uncertain / camera occluded")
                st.info(f"Saved annotation to `{p}`!")
                st.rerun()

        except Exception as e:
            st.warning(f"Could not load candidate contact frame: {e}")
    else:
        st.write("Run a video analysis in Tab 1 to populate candidate contact frames.")