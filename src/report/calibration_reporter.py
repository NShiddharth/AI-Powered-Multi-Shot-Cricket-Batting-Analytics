import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

CORE_METRIC_KEYS = [
    'front_elbow_angle',
    'spine_lean',
    'head_knee_alignment',
    'front_foot_direction',
    'follow_through_angle',
    'follow_through_quality',
    'head_stability',
    'shoulder_rotation',
    'hip_rotation',
    'stride_length'
]

class ShotCalibrationReporter:
    """
    Visualization & Report Generation Module for Empirical Reference Calibration.
    Supports Cover Drive, Straight Drive, Pull Shot, or arbitrary cricket shot classes.
    Generates:
      1. Distribution plots (P10-P90 interval plot, per-video bar chart, correlation matrix, radar profile)
      2. Standalone HTML calibration reports
      3. Optional PDF calibration reports
    Safely handles None/N/A values for metrics with low landmark confidence.
    """
    def __init__(self, reference_data, output_dir=None, shot_type=None):
        self.data = reference_data
        self.shot_type = (shot_type or reference_data.get('shot_type', 'cover_drive')).lower().replace(" ", "_")
        self.shot_display_name = self.shot_type.replace("_", " ").title()
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(f"output/reference_calibration/{self.shot_type}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_distribution_plots(self):
        """
        Creates comprehensive visualization suite:
          1. Metric Distributions & P10-P90 Reference Ranges
          2. Per-Video vs Reference Mean Comparisons
          3. Metric Correlation Matrix
          4. Radar Profile Chart
        Saves as {shot_type}_distribution.png
        """
        statistics = self.data.get('statistics', {})
        valid_videos = [v for v in self.data.get('per_video_data', []) if v['status'] in ('valid', 'warning')]

        metric_keys = [k for k in CORE_METRIC_KEYS if k in statistics]
        if not metric_keys:
            metric_keys = list(statistics.keys())

        if not metric_keys or not valid_videos:
            return None

        fig = plt.figure(figsize=(20, 16), dpi=120)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

        # 1. P10 - Mean - P90 Reference Interval Plot
        ax1 = fig.add_subplot(gs[0, :])
        metric_names = [statistics[k].get('name', k.replace('_', ' ').title()) for k in metric_keys]
        means = [statistics[k]['mean'] if statistics[k].get('mean') is not None else 0.0 for k in metric_keys]
        p10s = [statistics[k]['p10'] if statistics[k].get('p10') is not None else 0.0 for k in metric_keys]
        p90s = [statistics[k]['p90'] if statistics[k].get('p90') is not None else 0.0 for k in metric_keys]
        medians = [statistics[k]['median'] if statistics[k].get('median') is not None else 0.0 for k in metric_keys]

        y_pos = np.arange(len(metric_keys))

        for i in range(len(metric_keys)):
            ax1.plot([p10s[i], p90s[i]], [y_pos[i], y_pos[i]], color='#1f77b4', linewidth=3, alpha=0.7)
            ax1.plot(means[i], y_pos[i], 'ro', markersize=8, label='Mean' if i == 0 else "")
            ax1.plot(medians[i], y_pos[i], 's', color='green', markersize=6, label='Median' if i == 0 else "")

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(metric_names, fontsize=10, fontweight='bold')
        ax1.set_title(f"{self.shot_display_name} Biomechanical Metrics - Reference Ranges (P10 to P90, Mean & Median)", fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel("Metric Value", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='lower right')

        # 2. Per-Video Front Elbow Angle Comparison Bar Chart
        ax2 = fig.add_subplot(gs[1, 0])
        v_names = [v['video_id'][:13] for v in valid_videos]
        v_elbows = [v['metrics'].get('front_elbow_angle', 0.0) or 0.0 for v in valid_videos]
        ref_mean_elbow = statistics.get('front_elbow_angle', {}).get('mean', 0.0) or 0.0

        ax2.bar(v_names, v_elbows, color='#3498db', alpha=0.85, edgecolor='black')
        ax2.axhline(ref_mean_elbow, color='red', linestyle='--', linewidth=2, label=f"Ref Mean ({ref_mean_elbow:.1f}°)")
        if statistics.get('front_elbow_angle', {}).get('p10') is not None:
            ax2.axhline(statistics['front_elbow_angle']['p10'], color='orange', linestyle=':', linewidth=1.5, label="P10 Bound")
        if statistics.get('front_elbow_angle', {}).get('p90') is not None:
            ax2.axhline(statistics['front_elbow_angle']['p90'], color='orange', linestyle=':', linewidth=1.5, label="P90 Bound")
        ax2.set_title("Front Elbow Angle Across Calibration Videos", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Degrees (°)", fontsize=10)
        ax2.tick_params(axis='x', rotation=30, labelsize=9)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(fontsize=9)

        # 3. Metric Correlation Matrix Heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        data_matrix = []
        for v in valid_videos:
            row = [v['metrics'].get(k, 0.0) if v['metrics'].get(k) is not None else 0.0 for k in metric_keys]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)
        if data_matrix.shape[0] > 1:
            corr = np.corrcoef(data_matrix, rowvar=False)
            corr = np.nan_to_num(corr)
            im = ax3.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            short_names = [k.replace('_', ' ').title()[:10] for k in metric_keys]
            ax3.set_xticks(np.arange(len(metric_keys)))
            ax3.set_yticks(np.arange(len(metric_keys)))
            ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
            ax3.set_yticklabels(short_names, fontsize=8)
            ax3.set_title("Metric Correlation Heatmap", fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, "Insufficient samples for correlation", ha='center', va='center')

        # 4. Normalized Radar Chart Profile
        ax4 = fig.add_subplot(gs[2, :], polar=True)
        categories = [k.replace('_', ' ').title() for k in metric_keys]
        num_vars = len(categories)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        norm_means = []
        for k in metric_keys:
            st = statistics[k]
            if st.get('mean') is not None and st.get('max') is not None and st.get('min') is not None:
                span = max(1e-5, st['max'] - st['min'])
                norm_val = (st['mean'] - st['min']) / span
            else:
                norm_val = 0.0
            norm_means.append(norm_val)

        norm_means += norm_means[:1]

        ax4.plot(angles, norm_means, color='#2ecc71', linewidth=2.5, label=f'{self.shot_display_name} Reference Profile')
        ax4.fill(angles, norm_means, color='#2ecc71', alpha=0.25)
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=9, fontweight='bold')
        ax4.set_title(f"Normalized {self.shot_display_name} Biomechanical Radar Profile", fontsize=12, fontweight='bold', pad=15)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

        plot_path = self.output_dir / f"{self.shot_type}_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)

        # Generate dedicated publication-grade box plots suite
        try:
            from src.calibration.biomechanical_statistics import BiomechanicalStatisticsAnalyzer
            analyzer = BiomechanicalStatisticsAnalyzer(self.data, shot_type=self.shot_type)
            box_plot_path = self.output_dir / f"{self.shot_type}_boxplots.png"
            analyzer.plot_all_metric_boxplots(save_path=box_plot_path)
        except Exception as e:
            print(f"[!] Warning: Could not generate box plot in reporter: {e}")

        return str(plot_path)

    def generate_html_report(self):
        """
        Generates a styled, standalone HTML Calibration Report ({shot_type}_calibration_report.html).
        Format-safe handling for None/N/A values.
        """
        stats = self.data.get('statistics', {})
        per_video = self.data.get('per_video_data', [])
        outliers = self.data.get('outlier_diagnostics', {})

        valid_count = self.data.get('valid_videos_count', 0)
        total_count = self.data.get('total_videos_processed', 0)
        status = self.data.get('reference_status', 'provisional').title()

        def fmt(val, fmt_spec=".2f"):
            if val is None:
                return "N/A"
            try:
                return f"{val:{fmt_spec}}"
            except Exception:
                return str(val)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AthleteRise - {self.shot_display_name} Reference Calibration Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding: 30px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; border-bottom: 1px solid #ecf0f1; padding-bottom: 8px; }}
        .badge {{ display: inline-block; padding: 6px 12px; background: #e74c3c; color: white; border-radius: 4px; font-weight: bold; font-size: 14px; }}
        .badge-provisional {{ background: #f39c12; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin: 20px 0; }}
        .card {{ background: #f1f2f6; padding: 15px; border-radius: 6px; border-left: 4px solid #3498db; }}
        .card h4 {{ margin: 0 0 8px 0; color: #7f8c8d; font-size: 13px; text-transform: uppercase; }}
        .card p {{ margin: 0; font-size: 22px; font-weight: bold; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
        th, td {{ padding: 10px 12px; border: 1px solid #dfe6e9; text-align: left; }}
        th {{ background-color: #3498db; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .warning-box {{ background: #fff3cd; color: #856404; padding: 15px; border-radius: 6px; border-left: 5px solid #ffeba2; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 40px; color: #95a5a6; font-size: 12px; border-top: 1px solid #ecf0f1; padding-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏏 AthleteRise: {self.shot_display_name} Reference Calibration Report</h1>
        <p><strong>Shot Type:</strong> {self.shot_display_name} | <strong>Handedness:</strong> {self.data.get('handedness', 'mixed').upper()}</p>
        
        <div class="warning-box">
            <strong>⚠️ Reference Status Notice:</strong> <span class="badge badge-provisional">{status}</span><br>
            This calibration baseline was calculated from <strong>{valid_count}</strong> {self.shot_display_name} reference videos. It serves as a project-specific empirical reference dataset, not a universal medical standard.
        </div>

        <div class="grid">
            <div class="card">
                <h4>Total Videos Processed</h4>
                <p>{total_count}</p>
            </div>
            <div class="card">
                <h4>Valid Calibration Samples</h4>
                <p>{valid_count}</p>
            </div>
            <div class="card">
                <h4>Excluded Videos</h4>
                <p>{self.data.get('excluded_videos_count', 0)}</p>
            </div>
            <div class="card">
                <h4>Statistical Source</h4>
                <p>Empirical Reference</p>
            </div>
            <div class="card">
                <h4>Reference Range Method</h4>
                <p>P10 - P90</p>
            </div>
        </div>

        <div class="warning-box" style="background: #e8f4f8; color: #1b4f72; border-left: 5px solid #2980b9;">
            <strong>📌 Calibration Data Notice:</strong> Calculated from available reference/calibration data across all <strong>{valid_count}</strong> reference videos.
            These empirical statistical distributions (Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR, P10, P90) are derived purely from reference video distributions and remain separate from application scoring thresholds.
        </div>

        <h2>📊 Biomechanical Metric Statistical Summary (Empirical Reference Calibration)</h2>
        <p style="color: #7f8c8d; font-size: 13px;"><em>Each biomechanical metric is treated independently across all reference videos.</em></p>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Valid Samples</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Variance</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Q1 (P25)</th>
                    <th>Q3 (P75)</th>
                    <th>IQR</th>
                    <th>P10</th>
                    <th>P90</th>
                    <th>P10-P90 Range</th>
                </tr>
            </thead>
            <tbody>"""

        for k, v in stats.items():
            ref_rng = v.get('reference_range', {})
            low_str = fmt(ref_rng.get('lower'))
            high_str = fmt(ref_rng.get('upper'))
            rng_str = f"{low_str} - {high_str}" if low_str != "N/A" else "N/A"
            q1_v = v.get('q1', v.get('p25'))
            q3_v = v.get('q3', v.get('p75'))
            iqr_v = v.get('iqr')

            html_content += f"""
                <tr>
                    <td><strong>{v.get('name', k)}</strong></td>
                    <td>{v.get('valid_samples_count', 0)}</td>
                    <td>{fmt(v.get('mean'))}</td>
                    <td>{fmt(v.get('median'))}</td>
                    <td>{fmt(v.get('std'))}</td>
                    <td>{fmt(v.get('variance'))}</td>
                    <td>{fmt(v.get('min'))}</td>
                    <td>{fmt(v.get('max'))}</td>
                    <td>{fmt(q1_v)}</td>
                    <td>{fmt(q3_v)}</td>
                    <td>{fmt(iqr_v)}</td>
                    <td>{fmt(v.get('p10'))}</td>
                    <td>{fmt(v.get('p90'))}</td>
                    <td><strong>{rng_str}</strong></td>
                </tr>"""

        html_content += f"""
            </tbody>
        </table>

        <h2>📦 Empirical Biomechanical Metric Box Plots</h2>
        <p style="color: #7f8c8d; font-size: 13px;"><em>Visualizing central tendency (median & mean), spread (IQR box Q1–Q3), variability (whiskers), and potential outliers across reference videos.</em></p>
        <img src="{self.shot_type}_boxplots.png" alt="Biomechanical Metric Box Plots" onerror="this.style.display='none'">

        <h2>📈 Visualization Dashboard (Intervals & Correlations)</h2>
        <img src="{self.shot_type}_distribution.png" alt="Distribution Plots">

        <h2>📹 Video Dataset Quality & Validation</h2>
        <table>
            <thead>
                <tr>
                    <th>Video File</th>
                    <th>Status</th>
                    <th>Stance</th>
                    <th>Duration</th>
                    <th>FPS</th>
                    <th>Frames</th>
                    <th>Resolution</th>
                    <th>Pose Detection %</th>
                    <th>Contact Frame</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>"""

        for vid in per_video:
            st_color = "#2ecc71" if vid['status'] == 'valid' else ("#f39c12" if vid['status'] == 'warning' else "#e74c3c")
            html_content += f"""
                <tr>
                    <td><strong>{vid['filename']}</strong></td>
                    <td><span style="color: {st_color}; font-weight: bold;">{vid['status'].upper()}</span></td>
                    <td>{vid.get('handedness', 'right').capitalize()}</td>
                    <td>{vid['duration_sec']}s</td>
                    <td>{vid['fps']}</td>
                    <td>{vid['total_frames']}</td>
                    <td>{vid['resolution']}</td>
                    <td>{vid['pose_detection_pct']}%</td>
                    <td>{vid['contact_frame']}</td>
                    <td>{vid.get('exclusion_reason', 'OK')}</td>
                </tr>"""

        html_content += """
            </tbody>
        </table>

        <h2>🔍 Outlier Diagnostic Report (IQR transparent method)</h2>"""

        has_outliers = False
        for k, o_list in outliers.items():
            if o_list:
                has_outliers = True
                html_content += f"<h3>{stats.get(k, {}).get('name', k)} Outliers</h3><ul>"
                for item in o_list:
                    html_content += f"<li><strong>{item['filename']}</strong>: Raw Value = {item['raw_value']} ({item['reason']})</li>"
                html_content += "</ul>"

        if not has_outliers:
            html_content += "<p>No extreme statistical outliers detected across reference dataset.</p>"

        html_content += """
        <div class="footer">
            AthleteRise AI-Powered Cricket Analytics Platform &bull; Data-Driven Calibration Module
        </div>
    </div>
</body>
</html>"""

        report_path = self.output_dir / f"{self.shot_type}_calibration_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(report_path)

    def generate_pdf_report(self):
        """
        Attempts to generate a PDF calibration report if ReportLab or pdfkit is available.
        Provides clean fallback notice if PDF engine is not present.
        """
        pdf_path = self.output_dir / f"{self.shot_type}_calibration_report.pdf"
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 750, f"AthleteRise: {self.shot_display_name} Reference Calibration Report")
            c.setFont("Helvetica", 11)
            c.drawString(50, 725, f"Reference Status: {self.data.get('reference_status', 'provisional').title()}")
            c.drawString(50, 705, f"Valid Reference Videos: {self.data.get('valid_videos_count', 0)}")
            c.drawString(50, 685, f"Total Videos Processed: {self.data.get('total_videos_processed', 0)}")
            c.drawString(50, 665, f"Statistical reference saved to {self.shot_type}_reference.json")
            c.drawString(50, 645, f"Detailed HTML report available at {self.shot_type}_calibration_report.html")
            c.save()
            return str(pdf_path)
        except Exception:
            return None


class CoverDriveCalibrationReporter(ShotCalibrationReporter):
    """
    Backward-compatible subclass for Cover Drive Calibration.
    """
    def __init__(self, reference_data, output_dir="output/reference_calibration/cover_drive"):
        super().__init__(reference_data, output_dir=output_dir, shot_type="cover_drive")
