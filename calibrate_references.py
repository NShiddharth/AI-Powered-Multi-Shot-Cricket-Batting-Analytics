"""
AthleteRise - Reference Calibration Orchestrator
Recalibrates independent reference databases across all three shot classes:
  - Cover Drive
  - Straight Drive
  - Pull Shot
Generates clean JSON baselines and prints comparative diagnostics across shots.
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, Any

from src.calibration.base_calibration_engine import ShotCalibrationEngine, METRIC_KEYS, METRIC_DISPLAY_NAMES
from src.shot_analysis.shot_config import SHOT_COVER_DRIVE, SHOT_STRAIGHT_DRIVE, SHOT_PULL_SHOT, SUPPORTED_SHOTS

def run_all_calibrations() -> Dict[str, Any]:
    print("=" * 70)
    print("      ATHLETERISE — MULTI-SHOT REFERENCE RECALIBRATION")
    print("=" * 70)

    summary_results = {}
    diagnostics = {}

    for shot in SUPPORTED_SHOTS:
        print(f"\n[+] Calibrating Shot Class: {shot.upper()} ...")
        engine = ShotCalibrationEngine(shot_type=shot)
        
        # Verify video files in directory
        video_files = sorted(glob.glob(str(engine.reference_dir / "*.avi")) + glob.glob(str(engine.reference_dir / "*.mp4")))
        print(f"    Found {len(video_files)} reference videos in {engine.reference_dir}")
        for vf in video_files:
            print(f"      - {os.path.basename(vf)}")

        res = engine.run_calibration(video_files)

        valid_count = res.get('valid_videos_count', 0)
        total_count = res.get('total_videos_processed', 0)
        summary_results[shot] = {
            'valid': valid_count,
            'total': total_count,
            'status': f"{valid_count}/{total_count} valid"
        }
        diagnostics[shot] = res.get('statistics', {})
        print(f"    --> {shot.upper()} Calibration Complete: {valid_count}/{total_count} valid clips.")

    # Generate comparative diagnostic matrix
    print("\n" + "=" * 80)
    print("                 CALIBRATION BIOMECHANICAL DIAGNOSTICS")
    print("=" * 80)
    header = f"{'Metric':<25} | {'Cover Mean':<10} | {'Cover Std':<10} | {'Straight Mean':<13} | {'Straight Std':<12} | {'Pull Mean':<10} | {'Pull Std':<10}"
    print(header)
    print("-" * 80)

    report_lines = [header, "-" * 80]

    for key in METRIC_KEYS:
        c_stat = diagnostics.get(SHOT_COVER_DRIVE, {}).get(key, {})
        s_stat = diagnostics.get(SHOT_STRAIGHT_DRIVE, {}).get(key, {})
        p_stat = diagnostics.get(SHOT_PULL_SHOT, {}).get(key, {})

        c_m = f"{c_stat.get('mean', 0.0):.2f}" if c_stat.get('mean') is not None else "N/A"
        c_s = f"{c_stat.get('std', 0.0):.2f}" if c_stat.get('std') is not None else "N/A"
        s_m = f"{s_stat.get('mean', 0.0):.2f}" if s_stat.get('mean') is not None else "N/A"
        s_s = f"{s_stat.get('std', 0.0):.2f}" if s_stat.get('std') is not None else "N/A"
        p_m = f"{p_stat.get('mean', 0.0):.2f}" if p_stat.get('mean') is not None else "N/A"
        p_s = f"{p_stat.get('std', 0.0):.2f}" if p_stat.get('std') is not None else "N/A"

        row = f"{METRIC_DISPLAY_NAMES.get(key, key)[:25]:<25} | {c_m:<10} | {c_s:<10} | {s_m:<13} | {s_s:<12} | {p_m:<10} | {p_s:<10}"
        print(row)
        report_lines.append(row)

    print("=" * 80)

    # Save diagnostic matrix to file
    out_dir = Path("output/reference_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "calibration_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\nSummary:")
    for shot, info in summary_results.items():
        print(f"  - {shot.replace('_', ' ').title()}: {info['status']}")

    validate_calibrations()
    return summary_results

def validate_calibrations():
    """
    Step 17: Calibration Validation Report
    Inspects all reference calibration baselines and reports:
    - Class
    - Valid videos
    - Invalid videos (with reasons)
    - Mean, Median, Std, P10, P90
    - Feature availability
    """
    print("\n" + "=" * 80)
    print("           STEP 17 — CALIBRATION VALIDATION REPORT")
    print("=" * 80)
    
    validation_lines = ["=" * 80, "           STEP 17 — CALIBRATION VALIDATION REPORT", "=" * 80]

    for shot in SUPPORTED_SHOTS:
        ref_file = Path(f"output/reference_calibration/{shot}/{shot}_reference.json")
        if not ref_file.exists():
            msg = f"\n{shot.replace('_', ' ').title()}: Reference JSON not found at {ref_file}"
            print(msg)
            validation_lines.append(msg)
            continue

        with open(ref_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_vid = data.get("per_video_data", [])
        valid_vids = [v for v in per_vid if v.get("status") in ("valid", "warning")]
        invalid_vids = [v for v in per_vid if v.get("status") == "excluded"]

        class_title = shot.replace('_', ' ').title()
        header = f"\n--- {class_title} ---"
        stat_valid = f"Valid videos: {len(valid_vids)}/{len(per_vid)}"
        print(header)
        print(stat_valid)
        validation_lines.extend([header, stat_valid])

        if invalid_vids:
            print("Invalid videos:")
            validation_lines.append("Invalid videos:")
            for iv in invalid_vids:
                reason = iv.get("exclusion_reason", "Unknown reason")
                fname = iv.get("filename", "unknown")
                inv_line = f"  - {fname} — {reason}"
                print(inv_line)
                validation_lines.append(inv_line)
        else:
            no_fail = "Invalid videos: 0 (All clips met valid pose tracking thresholds)"
            print(no_fail)
            validation_lines.append(no_fail)

        stats = data.get("statistics", {})
        metric_table_hdr = f"\n{'Metric':<25} | {'Mean':<7} | {'Median':<7} | {'Std':<7} | {'Var':<7} | {'Min':<7} | {'Max':<7} | {'Q1':<7} | {'Q3':<7} | {'IQR':<7} | {'P10':<7} | {'P90':<7}"
        print(metric_table_hdr)
        print("-" * 115)
        validation_lines.extend([
            f"\nData Source: Calculated from available reference/calibration data across all {len(valid_vids)} reference videos",
            metric_table_hdr,
            "-" * 115
        ])

        for k, v in stats.items():
            m_name = v.get("name", k)[:25]
            mean_v = f"{v.get('mean', 0.0):.2f}" if v.get('mean') is not None else "N/A"
            med_v = f"{v.get('median', 0.0):.2f}" if v.get('median') is not None else "N/A"
            std_v = f"{v.get('std', 0.0):.2f}" if v.get('std') is not None else "N/A"
            var_v = f"{v.get('variance', 0.0):.2f}" if v.get('variance') is not None else "N/A"
            min_v = f"{v.get('min', 0.0):.2f}" if v.get('min') is not None else "N/A"
            max_v = f"{v.get('max', 0.0):.2f}" if v.get('max') is not None else "N/A"
            q1_v = f"{v.get('q1', v.get('p25', 0.0)):.2f}" if (v.get('q1') or v.get('p25')) is not None else "N/A"
            q3_v = f"{v.get('q3', v.get('p75', 0.0)):.2f}" if (v.get('q3') or v.get('p75')) is not None else "N/A"
            iqr_v = f"{v.get('iqr', 0.0):.2f}" if v.get('iqr') is not None else "N/A"
            p10_v = f"{v.get('p10', 0.0):.2f}" if v.get('p10') is not None else "N/A"
            p90_v = f"{v.get('p90', 0.0):.2f}" if v.get('p90') is not None else "N/A"

            row = f"{m_name:<25} | {mean_v:<7} | {med_v:<7} | {std_v:<7} | {var_v:<7} | {min_v:<7} | {max_v:<7} | {q1_v:<7} | {q3_v:<7} | {iqr_v:<7} | {p10_v:<7} | {p90_v:<7}"
            print(row)
            validation_lines.append(row)

        # Generate box plot for this shot if not already generated
        try:
            from src.calibration.biomechanical_statistics import BiomechanicalStatisticsAnalyzer
            analyzer = BiomechanicalStatisticsAnalyzer(data, shot_type=shot)
            box_plot_path = Path(f"output/reference_calibration/{shot}/{shot}_boxplots.png")
            analyzer.plot_all_metric_boxplots(save_path=box_plot_path)
            box_msg = f"[+] Generated box plots: {box_plot_path}"
            print(box_msg)
            validation_lines.append(box_msg)
        except Exception as e:
            print(f"[!] Warning: Could not generate box plot for {shot}: {e}")

        # Generate distribution plots and HTML report for this shot
        try:
            from src.report.calibration_reporter import ShotCalibrationReporter
            reporter = ShotCalibrationReporter(data, output_dir=f"output/reference_calibration/{shot}", shot_type=shot)
            dist_path = reporter.generate_distribution_plots()
            html_path = reporter.generate_html_report()
            dist_msg = f"[+] Generated distribution plots: {dist_path}"
            html_msg = f"[+] Generated HTML report: {html_path}"
            print(dist_msg)
            print(html_msg)
            validation_lines.extend([dist_msg, html_msg])
        except Exception as e:
            print(f"[!] Warning: Could not generate distribution plots or HTML report for {shot}: {e}")

    report_path = Path("output/reference_calibration/calibration_validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(validation_lines))
    print(f"\n[+] Validation report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing calibrations")
    args = parser.parse_args()

    if args.validate_only:
        validate_calibrations()
    else:
        run_all_calibrations()
