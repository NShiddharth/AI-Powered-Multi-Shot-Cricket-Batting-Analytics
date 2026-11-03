"""
AthleteRise - Biomechanical Metrics Statistical Analysis & Box Plot Visualizer
Treats each batting/cover-drive biomechanical metric independently.

Computes comprehensive empirical statistics across reference/calibration videos:
- Mean
- Median
- Std Dev
- Variance
- Min
- Max
- Q1 (25th Percentile)
- Q3 (75th Percentile)
- IQR (Interquartile Range = Q3 - Q1)
- P10 (10th Percentile)
- P90 (90th Percentile)
- Potential Outliers (Tukey IQR method: < Q1 - 1.5*IQR or > Q3 + 1.5*IQR)

Provides publication-grade box plots visualizing:
- Central tendency (Median line, Mean marker)
- Spread (IQR box from Q1 to Q3)
- Variability (Whiskers: min/max or 1.5*IQR bounds)
- Potential outliers (Individual flier points)
- Raw reference video data points overlay (jittered points)

IMPORTANT: These empirical statistical distributions are derived strictly from
the reference/calibration dataset across all reference videos and remain completely
separate from any heuristic scoring thresholds.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Default canonical metric names and user-friendly display labels
DEFAULT_METRIC_DISPLAY_NAMES = {
    'front_elbow_angle': 'Front Elbow Angle (°)',
    'head_knee_alignment': 'Head-Knee Alignment (norm)',
    'front_foot_direction': 'Front Foot Direction (°)',
    'spine_lean': 'Spine Lean Tilt (°)',
    'shoulder_rotation': 'Shoulder Rotation (°)',
    'hip_rotation': 'Hip Rotation (°)',
    'stride_length': 'Stride Length (norm)',
    'head_stability': 'Head Stability (norm)',
    'follow_through_angle': 'Follow-Through Angle (°)',
    'follow_through_quality': 'Follow-Through Quality (0-100)',
    'rotary_shoulder_diff': 'Rotary Shoulder Diff (°)',
    'wrist_horizontal_travel': 'Wrist Horizontal Travel (norm)',
    'wrist_vertical_travel': 'Wrist Vertical Travel (norm)'
}

# Unit labels for metrics
METRIC_UNITS = {
    'front_elbow_angle': '°',
    'head_knee_alignment': 'norm',
    'front_foot_direction': '°',
    'spine_lean': '°',
    'shoulder_rotation': '°',
    'hip_rotation': '°',
    'stride_length': 'norm',
    'head_stability': 'norm',
    'follow_through_angle': '°',
    'follow_through_quality': '/100',
    'rotary_shoulder_diff': '°',
    'wrist_horizontal_travel': 'norm',
    'wrist_vertical_travel': 'norm'
}

DATASET_LABEL = "Calculated from available reference/calibration data across all reference videos"


def get_metric_display_name(metric_key: str) -> str:
    """Returns friendly display name for a biomechanical metric key."""
    if metric_key in DEFAULT_METRIC_DISPLAY_NAMES:
        return DEFAULT_METRIC_DISPLAY_NAMES[metric_key]
    # Format unknown numerical metrics gracefully
    clean = metric_key.replace("_", " ").title()
    return clean


def compute_metric_statistics(values: List[Union[float, int]],
                              metric_key: str = "",
                              metric_name: Optional[str] = None,
                              video_identifiers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Computes all 11 required statistical descriptors for a single biomechanical metric
    treated completely independently:
    - Mean
    - Median
    - Std Dev
    - Variance
    - Min
    - Max
    - Q1 (25th Percentile)
    - Q3 (75th Percentile)
    - IQR (Q3 - Q1)
    - P10 (10th Percentile)
    - P90 (90th Percentile)
    plus outlier diagnostics and raw values.
    """
    clean_pairs = []
    if video_identifiers and len(video_identifiers) == len(values):
        for vid, v in zip(video_identifiers, values):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                try:
                    clean_pairs.append((vid, float(v)))
                except (ValueError, TypeError):
                    pass
    else:
        for idx, v in enumerate(values):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                try:
                    clean_pairs.append((f"ref_{idx+1}", float(v)))
                except (ValueError, TypeError):
                    pass

    display_name = metric_name or get_metric_display_name(metric_key)

    if not clean_pairs:
        return {
            'metric_key': metric_key,
            'metric_name': display_name,
            'data_source_label': DATASET_LABEL,
            'valid_samples_count': 0,
            'mean': None,
            'median': None,
            'std_dev': None,
            'variance': None,
            'min': None,
            'max': None,
            'q1': None,
            'q3': None,
            'iqr': None,
            'p10': None,
            'p90': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'outliers': [],
            'raw_values': [],
            'per_video_values': {}
        }

    raw_nums = np.array([p[1] for p in clean_pairs], dtype=float)
    valid_count = len(raw_nums)

    mean_val = float(np.mean(raw_nums))
    median_val = float(np.median(raw_nums))
    std_val = float(np.std(raw_nums))
    var_val = float(np.var(raw_nums))
    min_val = float(np.min(raw_nums))
    max_val = float(np.max(raw_nums))

    p10_val = float(np.percentile(raw_nums, 10))
    q1_val = float(np.percentile(raw_nums, 25))
    q3_val = float(np.percentile(raw_nums, 75))
    p90_val = float(np.percentile(raw_nums, 90))
    iqr_val = float(q3_val - q1_val)

    lower_bound = q1_val - 1.5 * iqr_val
    upper_bound = q3_val + 1.5 * iqr_val

    outliers = []
    per_video = {}
    for vid, val in clean_pairs:
        per_video[vid] = round(val, 4)
        if val < lower_bound or val > upper_bound:
            outliers.append({
                'video_id': vid,
                'value': round(val, 4),
                'bound_breached': 'lower' if val < lower_bound else 'upper',
                'reason': f"Outside IQR whisker bounds [{round(lower_bound, 3)}, {round(upper_bound, 3)}]"
            })

    return {
        'metric_key': metric_key,
        'metric_name': display_name,
        'data_source_label': DATASET_LABEL,
        'valid_samples_count': valid_count,
        'mean': round(mean_val, 4),
        'median': round(median_val, 4),
        'std_dev': round(std_val, 4),
        'variance': round(var_val, 4),
        'min': round(min_val, 4),
        'max': round(max_val, 4),
        'q1': round(q1_val, 4),
        'q3': round(q3_val, 4),
        'iqr': round(iqr_val, 4),
        'p10': round(p10_val, 4),
        'p90': round(p90_val, 4),
        # Backward compatibility aliases
        'p25': round(q1_val, 4),
        'p50': round(median_val, 4),
        'p75': round(q3_val, 4),
        'whisker_lower': round(lower_bound, 4),
        'whisker_upper': round(upper_bound, 4),
        'outliers': outliers,
        'raw_values': [round(float(x), 4) for x in raw_nums],
        'per_video_values': per_video
    }


class BiomechanicalStatisticsAnalyzer:
    """
    Independent Statistical Analyzer and Box Plot Generator for Batting Biomechanics.
    Processes reference videos / calibration datasets across all supported metrics.
    """

    def __init__(self, reference_data_or_path: Union[str, Path, Dict[str, Any]], shot_type: str = "cover_drive"):
        self.shot_type = shot_type.lower().replace(" ", "_")
        self.raw_data = self._load_data(reference_data_or_path)
        self.per_video_data = self.raw_data.get('per_video_data', [])
        self.valid_videos = [v for v in self.per_video_data if v.get('status') in ('valid', 'warning')]
        if not self.valid_videos and self.per_video_data:
            self.valid_videos = self.per_video_data

        # Discover all available numerical biomechanical metrics
        self.metric_keys = self._discover_all_numerical_metrics()
        self.statistics = self._compute_all_statistics()

    def _load_data(self, source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(source, dict):
            return source
        p = Path(source)
        if not p.exists():
            # Attempt to find standard shot reference file
            default_p = Path(f"output/reference_calibration/{self.shot_type}/{self.shot_type}_reference.json")
            if default_p.exists():
                p = default_p
            else:
                return {'statistics': {}, 'per_video_data': []}
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _discover_all_numerical_metrics(self) -> List[str]:
        """
        Dynamically finds all numerical biomechanical metrics present in:
        1. Existing statistics dict
        2. Per-video metric dictionaries
        Guarantees coverage of Elbow Angle, Head-Knee Distance, Foot Angle, Spine Lean,
        and ANY other numerical biomechanical metric present in the dataset.
        """
        discovered = []

        # Start with primary metrics in priority order if present
        priority_keys = [
            'front_elbow_angle',
            'head_knee_alignment',
            'front_foot_direction',
            'spine_lean',
            'shoulder_rotation',
            'hip_rotation',
            'stride_length',
            'head_stability',
            'follow_through_angle',
            'follow_through_quality',
            'rotary_shoulder_diff',
            'wrist_horizontal_travel',
            'wrist_vertical_travel'
        ]

        seen = set()

        # Check existing statistics block
        stats = self.raw_data.get('statistics', {})
        for k in stats:
            if k not in seen:
                discovered.append(k)
                seen.add(k)

        # Check per-video records for any other numerical metrics
        for vid in self.valid_videos:
            metrics_dict = vid.get('metrics', {})
            for k, val in metrics_dict.items():
                if k not in seen and val is not None and isinstance(val, (int, float)):
                    discovered.append(k)
                    seen.add(k)

        # Re-sort to put priority keys first while preserving any novel metric keys
        sorted_keys = [k for k in priority_keys if k in seen]
        for k in discovered:
            if k not in sorted_keys:
                sorted_keys.append(k)

        return sorted_keys

    def _compute_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Computes independent statistical analysis for every discovered metric."""
        stats_results = {}
        for mk in self.metric_keys:
            raw_vals = []
            vid_ids = []
            for vid in self.valid_videos:
                v_metrics = vid.get('metrics', {})
                if mk in v_metrics and v_metrics[mk] is not None:
                    raw_vals.append(v_metrics[mk])
                    vid_ids.append(vid.get('video_id') or vid.get('filename') or "ref_clip")

            # Fallback to existing statistics if raw per-video was empty
            if not raw_vals and mk in self.raw_data.get('statistics', {}):
                cached = self.raw_data['statistics'][mk]
                if cached.get('mean') is not None:
                    q1 = cached.get('q1') or cached.get('p25')
                    q3 = cached.get('q3') or cached.get('p75')
                    iqr = cached.get('iqr') or ((q3 - q1) if (q3 is not None and q1 is not None) else None)
                    stats_results[mk] = {
                        'metric_key': mk,
                        'metric_name': cached.get('name', get_metric_display_name(mk)),
                        'data_source_label': DATASET_LABEL,
                        'valid_samples_count': cached.get('valid_samples_count', len(self.valid_videos)),
                        'mean': cached.get('mean'),
                        'median': cached.get('median'),
                        'std_dev': cached.get('std'),
                        'variance': cached.get('variance'),
                        'min': cached.get('min'),
                        'max': cached.get('max'),
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'p10': cached.get('p10'),
                        'p90': cached.get('p90'),
                        'p25': q1,
                        'p50': cached.get('median'),
                        'p75': q3,
                        'outliers': cached.get('outliers', []),
                        'raw_values': [],
                        'per_video_values': {}
                    }
                    continue

            stat = compute_metric_statistics(
                values=raw_vals,
                metric_key=mk,
                metric_name=get_metric_display_name(mk),
                video_identifiers=vid_ids
            )
            stats_results[mk] = stat

        return stats_results

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Builds the complete comparative table allowing comparison of biomechanical metrics using:
        Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR, P10 and P90.
        """
        rows = []
        for k in self.metric_keys:
            st = self.statistics.get(k, {})
            rows.append({
                'Biomechanical Metric': st.get('metric_name', get_metric_display_name(k)),
                'Metric Key': k,
                'Valid Videos': st.get('valid_samples_count', 0),
                'Mean': st.get('mean'),
                'Median': st.get('median'),
                'Std Dev': st.get('std_dev'),
                'Variance': st.get('variance'),
                'Min': st.get('min'),
                'Max': st.get('max'),
                'Q1': st.get('q1'),
                'Q3': st.get('q3'),
                'IQR': st.get('iqr'),
                'P10': st.get('p10'),
                'P90': st.get('p90'),
                'Outliers Count': len(st.get('outliers', []))
            })
        df = pd.DataFrame(rows)
        return df

    def plot_all_metric_boxplots(self, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Generates a comprehensive multi-panel figure displaying a box plot
        for EACH biomechanical metric in the dataset.
        Shows:
        - Central tendency (Median line, Mean diamond marker)
        - Spread (IQR box from Q1 to Q3)
        - Variability (Whiskers: min to max / 1.5*IQR)
        - Potential outliers (Individual flier points)
        - Raw data overlay with jitter
        """
        valid_keys = [k for k in self.metric_keys if len(self.statistics[k].get('raw_values', [])) > 0]
        if not valid_keys:
            valid_keys = self.metric_keys

        n_metrics = len(valid_keys)
        if n_metrics == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No reference metric data available", ha='center', va='center')
            return fig

        n_cols = 3 if n_metrics > 6 else (2 if n_metrics > 1 else 1)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(n_cols * 5.2, n_rows * 4.2), dpi=120)
        shot_title = self.shot_type.replace('_', ' ').title()

        fig.suptitle(
            f"AthleteRise - {shot_title} Reference Calibration: Biomechanical Metrics Box Plots\n"
            f"{DATASET_LABEL} (N = {len(self.valid_videos)} reference videos)",
            fontsize=14, fontweight='bold', y=0.99
        )

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.48, wspace=0.38)

        for idx, key in enumerate(valid_keys):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            st = self.statistics[key]
            vals = st.get('raw_values', [])

            if not vals:
                # Reconstruct points from summary if raw missing
                vals = [st.get('min', 0), st.get('q1', 0), st.get('median', 0), st.get('q3', 0), st.get('max', 0)]
                vals = [v for v in vals if v is not None]

            # Box plot rendering
            bp = ax.boxplot(
                vals,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                widths=0.45,
                flierprops=dict(marker='o', markerfacecolor='#e74c3c', markeredgecolor='black', markersize=7, alpha=0.9),
                boxprops=dict(facecolor='#e8f4f8', color='#1f77b4', linewidth=1.8),
                whiskerprops=dict(color='#2c3e50', linewidth=1.5, linestyle='--'),
                capprops=dict(color='#2c3e50', linewidth=2.0),
                medianprops=dict(color='#e74c3c', linewidth=2.5),
                meanprops=dict(marker='D', markerfacecolor='#27ae60', markeredgecolor='black', markersize=6)
            )

            # Jitter overlay for raw data points
            jitter = np.random.normal(1.0, 0.04, size=len(vals)) if len(vals) > 1 else [1.0]
            ax.scatter(jitter, vals, color='#2980b9', alpha=0.6, s=35, zorder=3, edgecolors='white', linewidth=0.5)

            # Mark P10 and P90 with subtle horizontal lines
            if st.get('p10') is not None and st.get('p90') is not None:
                ax.axhline(st['p10'], color='#f39c12', linestyle=':', linewidth=1.2, alpha=0.8)
                ax.axhline(st['p90'], color='#f39c12', linestyle=':', linewidth=1.2, alpha=0.8)

            m_name = st.get('metric_name', key)
            unit = METRIC_UNITS.get(key, '')
            unit_str = f" ({unit})" if unit else ""
            ax.set_title(f"{m_name}", fontsize=11, fontweight='bold', pad=8)
            ax.set_ylabel(f"Value{unit_str}", fontsize=9)
            ax.set_xticks([1])
            ax.set_xticklabels([f"Ref Dist (n={st.get('valid_samples_count', len(vals))})"], fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.4)

            # Annotate key stats
            med_str = f"Median: {st.get('median')}"
            mean_str = f"Mean: {st.get('mean')}"
            iqr_str = f"IQR: {st.get('iqr')}"
            n_out = len(st.get('outliers', []))
            out_str = f"Outliers: {n_out}" if n_out > 0 else "No Outliers"

            stat_note = f"{med_str} | {mean_str}\n{iqr_str} | {out_str}"
            ax.text(0.97, 0.03, stat_note, transform=ax.transAxes,
                    fontsize=7.5, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))

        # Add global legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#e74c3c', lw=2.5, label='Median (Central Tendency)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#27ae60', markeredgecolor='k', markersize=7, label='Mean'),
            Line2D([0], [0], color='#1f77b4', lw=6, alpha=0.4, label='IQR Spread (Q1–Q3)'),
            Line2D([0], [0], color='#2c3e50', lw=1.5, linestyle='--', label='Variability (Whiskers)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markeredgecolor='k', markersize=7, label='Potential Outliers'),
            Line2D([0], [0], color='#f39c12', lw=1.2, linestyle=':', label='P10 & P90 Bounds')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.005), fontsize=9)

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=130, bbox_inches='tight')

        return fig

    def plot_single_metric_boxplot(self, metric_key: str, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Generates an in-depth, dedicated single-metric box plot highlighting:
        - Central tendency (Median line, Mean marker)
        - Spread (IQR box: Q1 to Q3)
        - Variability (Whiskers: min, max, 1.5*IQR bounds)
        - Potential outliers (annotated with clip names)
        - Empirical reference data distribution
        """
        st = self.statistics.get(metric_key, {})
        m_name = st.get('metric_name', get_metric_display_name(metric_key))
        unit = METRIC_UNITS.get(metric_key, '')
        unit_str = f" ({unit})" if unit else ""

        vals = st.get('raw_values', [])
        if not vals:
            vals = [st.get('min', 0), st.get('q1', 0), st.get('median', 0), st.get('q3', 0), st.get('max', 0)]
            vals = [v for v in vals if v is not None]

        fig, (ax_box, ax_details) = plt.subplots(1, 2, figsize=(13, 6), dpi=120, gridspec_kw={'width_ratios': [1.2, 1.0]})

        # Subplot 1: The High-Resolution Box Plot
        bp = ax_box.boxplot(
            vals,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            widths=0.45,
            flierprops=dict(marker='o', markerfacecolor='#e74c3c', markeredgecolor='black', markersize=8, alpha=0.9),
            boxprops=dict(facecolor='#d4e6f1', color='#2980b9', linewidth=2.0),
            whiskerprops=dict(color='#2c3e50', linewidth=1.8, linestyle='--'),
            capprops=dict(color='#2c3e50', linewidth=2.2),
            medianprops=dict(color='#c0392b', linewidth=3.0),
            meanprops=dict(marker='D', markerfacecolor='#27ae60', markeredgecolor='black', markersize=8)
        )

        # Overlay individual reference video points with jitter & annotations
        np.random.seed(42)
        jitter = np.random.normal(1.0, 0.05, size=len(vals)) if len(vals) > 1 else [1.0]
        ax_box.scatter(jitter, vals, color='#1b4f72', alpha=0.75, s=65, zorder=4, edgecolors='white', linewidth=1.0)

        # Highlight P10 and P90 intervals
        if st.get('p10') is not None and st.get('p90') is not None:
            ax_box.axhline(st['p10'], color='#e67e22', linestyle=':', linewidth=1.8, label=f"P10 ({st['p10']})")
            ax_box.axhline(st['p90'], color='#e67e22', linestyle=':', linewidth=1.8, label=f"P90 ({st['p90']})")

        # Label Q1, Median, Q3, Mean directly on box plot
        if st.get('q1') is not None:
            ax_box.text(1.28, st['q1'], f"Q1 = {st['q1']}", color='#2980b9', fontsize=9, fontweight='bold', va='center')
        if st.get('median') is not None:
            ax_box.text(1.28, st['median'], f"Median = {st['median']}", color='#c0392b', fontsize=9, fontweight='bold', va='center')
        if st.get('mean') is not None:
            ax_box.text(0.72, st['mean'], f"Mean = {st['mean']}", color='#27ae60', fontsize=9, fontweight='bold', va='center', ha='right')
        if st.get('q3') is not None:
            ax_box.text(1.28, st['q3'], f"Q3 = {st['q3']}", color='#2980b9', fontsize=9, fontweight='bold', va='center')

        # Highlight and label any outliers
        for out in st.get('outliers', []):
            ax_box.annotate(
                f"{out['video_id']}\n({out['value']})",
                xy=(1.0, out['value']),
                xytext=(1.35, out['value']),
                arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=1, headwidth=5),
                fontsize=8, color='#c0392b', fontweight='bold'
            )

        ax_box.set_title(f"{m_name} — Empirical Box Plot", fontsize=12, fontweight='bold', pad=10)
        ax_box.set_ylabel(f"Biomechanical Measurement{unit_str}", fontsize=10, fontweight='bold')
        ax_box.set_xticks([1])
        ax_box.set_xticklabels([f"Reference Videos (N = {st.get('valid_samples_count', len(vals))})"], fontsize=10)
        ax_box.grid(axis='y', linestyle='--', alpha=0.5)

        # Subplot 2: Analytical Summary Card & Distribution Diagnostics
        ax_details.axis('off')
        shot_title = self.shot_type.replace('_', ' ').title()

        out_count = len(st.get('outliers', []))
        out_badge = f"{out_count} Detected" if out_count > 0 else "None (Clean)"

        summary_text = (
            f"=====================================\n"
            f"  {m_name.upper()}\n"
            f"  Empirical Calibration Baseline ({shot_title})\n"
            f"=====================================\n\n"
            f"[DATA SOURCE]\n"
            f"   {DATASET_LABEL}\n"
            f"   Total Reference Samples: {st.get('valid_samples_count', 0)} videos\n\n"
            f"[CENTRAL TENDENCY]\n"
            f"   * Median (P50) : {st.get('median')} {unit}\n"
            f"   * Mean         : {st.get('mean')} {unit}\n\n"
            f"[SPREAD & DISPERSION]\n"
            f"   * Q1 (P25)     : {st.get('q1')} {unit}\n"
            f"   * Q3 (P75)     : {st.get('q3')} {unit}\n"
            f"   * IQR (Q3 - Q1): {st.get('iqr')} {unit}\n\n"
            f"[VARIABILITY & EXTREMES]\n"
            f"   * Standard Dev : {st.get('std_dev')} {unit}\n"
            f"   * Variance     : {st.get('variance')}\n"
            f"   * Minimum      : {st.get('min')} {unit}\n"
            f"   * Maximum      : {st.get('max')} {unit}\n"
            f"   * P10 (10th %) : {st.get('p10')} {unit}\n"
            f"   * P90 (90th %) : {st.get('p90')} {unit}\n\n"
            f"[POTENTIAL OUTLIERS]\n"
            f"   * Outlier Count: {out_badge}\n"
            f"   * Method       : Tukey IQR [Q1 - 1.5*IQR, Q3 + 1.5*IQR]\n\n"
            f"[NOTE]\n"
            f"   These statistics reflect empirical video reference data.\n"
            f"   Scoring thresholds remain distinct and separate."
        )

        ax_details.text(
            0.05, 0.95, summary_text,
            transform=ax_details.transAxes,
            fontsize=9.5,
            family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#bdc3c7', linewidth=1.5)
        )

        plt.suptitle(
            f"AthleteRise Biomechanical Statistical Analysis — {m_name}\n"
            f"{DATASET_LABEL}",
            fontsize=13, fontweight='bold', y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=130, bbox_inches='tight')

        return fig
