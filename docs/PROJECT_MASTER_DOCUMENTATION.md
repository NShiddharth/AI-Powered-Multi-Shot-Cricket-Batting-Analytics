# AthleteRise: AI-Powered Multi-Shot Cricket Batting Analytics
## Comprehensive Technical Project Documentation & Master System Reference

---

### Project Information
- **Official Project Title:** AI-Powered Cricket Analytics (Platform Designation: **AthleteRise**)
- **Target Domain:** Monocular Computer Vision, Batting Kinematics, Multi-Shot Classification, Bat–Ball Contact Detection, and Empirical Reference Calibration
- **Repository Designation:** `AI-Powered-Cricket-Analytics-main`
- **Current Operational Status:** Fully Functional Multi-Shot Batting Analytics Platform (Cover Drive, Straight Drive, Pull Shot) with Empirical Reference Baselines ($N=30$, 10 clips/class), Modular Bat/Ball Contact Estimation, and 3-Tab Streamlit Dashboard
- **Documentation Status:** Single Authoritative Technical Reference (Supersedes all previous documentation versions)

---

# 1. Project Title

**AthleteRise: AI-Powered Cricket Analytics**  
*A Monocular Computer Vision and Biomechanical Analysis Engine for Multi-Shot Batting Technique Evaluation and Empirical Calibration.*

---

# 2. Executive Summary

**AthleteRise** is an automated, markerless computer-vision system designed to quantify, evaluate, and provide objective diagnostic feedback on cricket batting stroke execution using standard monocular 2D video (smartphones, consumer cameras, or broadcast feeds). 

In traditional sports training, biomechanical coaching relies almost exclusively on naked-eye observation or subjective qualitative impressions. Because human perceptual visual latency is roughly $100 - 150\text{ ms}$, while a competitive cricket stroke takes place within $300 - 500\text{ ms}$, coaches cannot consistently perceive micro-kinematic anomalies—such as subtle lead knee collapse, improper head-knee vertical alignment, or premature shoulder uncoiling. Meanwhile, commercial optical motion capture laboratories (e.g., multi-camera Vicon or Qualisys systems) cost upwards of $\$50,000$, require body-attached retro-reflective markers, and cannot be operated outdoors on live turf wickets.

AthleteRise addresses this gap by converting ordinary monocular video into scale-invariant, objective biomechanical telemetry through a five-stage architecture:
1. **Ingestion & Lighting Preprocessing:** Normalizes outdoor lighting variability using Contrast Limited Adaptive Histogram Equalization (CLAHE) on the Lightness channel in the LAB color space.
2. **Pose Tracking & Keypoint Extraction:** Ingests frames via Google MediaPipe Pose (leveraging the `pose_landmarker_lite.task` Tasks API with legacy solution fallback and optical-flow propagation) to track 15 essential anatomical keypoints.
3. **Data-Driven Shot Classification:** Extracts a 12-dimensional spatio-temporal kinematic feature vector and classifies the executed shot across three primary batting strokes (**Cover Drive**, **Straight Drive**, **Pull Shot**) or safely defaults to **Unknown**, using pooled-variance standardized Euclidean prototype distance and softmax confidence mapping.
4. **Bat–Ball Contact Detection:** Integrates dedicated bat blade segmentation and ball motion/color tracking to evaluate multi-signal physical evidence (spatial proximity, temporal alignment, relative velocity, ball trajectory deflection, and bat continuity), distinguishing the **Visual Contact Frame** from the **Biomechanical Pose Impact Estimate**.
5. **Continuous Distribution-Based Scoring:** Evaluates technique across five functional categories (Footwork, Head Position, Swing Control, Balance, and Follow-Through) using continuous percentile distribution curves ($P_{10} - P_{90}$) derived empirically from dedicated reference libraries ($N=10$ proficient videos per shot class), with dynamic occlusion weight renormalization to ensure fair assessment under partial camera occlusion.

---

# 3. Problem Statement

### 3.1 Perceptual Latency in Human Coaching
A standard cricket delivery takes between $400\text{ ms}$ and $600\text{ ms}$ from bowler release to the batter's contact point. The batsman's downswing acceleration occurs in less than $200\text{ ms}$. Because the human visual system requires approximately $100 - 150\text{ ms}$ to process and integrate sensory information, human coaching observation is constrained by visual persistence limitations. Coaches are forced to rely on post-impact ball trajectory rather than micro-biomechanical kinematic alignment during the impact window.

### 3.2 Subjectivity and Inconsistency
Coaching commentary is notoriously qualitative (e.g., *"lean into the drive,"* *"keep your head still,"* or *"watch the ball closely"*). Different coaches offer conflicting feedback for identical strokes, and athletes lack measurable metric baselines to monitor technical refinement across practice sessions.

### 3.3 Hardware Constraints of Laboratory Motion Capture
Traditional kinematic motion capture requires specialized multi-camera synchronized infrared cameras and retro-reflective body markers. These marker-based setups:
- Alter an athlete's natural swing mechanics due to physical encumbrance.
- Cannot be deployed outdoors due to solar infrared interference.
- Require high-end computing and financial investments prohibitive to grassroots academies.

### 3.4 Limitations of Monocular Video Analysis
Existing video-based sports applications typically suffer from:
- **Depth ambiguity** inherent in 2D perspective projections.
- **Occlusion** where the batter's own torso or bat obscures rear limbs.
- **Extreme lighting variations** across outdoor pitch environments (stadium floodlights, harsh sunlight, shadow lines).
- **Rigid rule-based heuristics** that penalize natural anatomical variance among different athletes.

---

# 4. Motivation

The primary motivation behind AthleteRise is the democratisation of high-performance sports biomechanics. By transforming standard consumer 2D video into calibrated kinematic metrics, AthleteRise enables:
- **Grassroots Accessibility:** Remote athletes and academies can obtain scientific biomechanical feedback using a standard mobile phone camera.
- **Empirical Grounding:** Rather than forcing every athlete to mimic the static geometry of a single professional champion, the system calibrates against an empirical statistical reference distribution ($P_{10} - P_{90}$), accommodating natural anatomical diversity while enforcing fundamental biomechanical efficiency.
- **Multi-Shot Scalability:** Athletes execute diverse strokes depending on bowling length and line; the system autonomously identifies whether a shot is an off-side front-foot drive, a vertical straight drive, or a back-foot cross-bat pull shot, tailoring evaluation baselines accordingly.
- **Explainable Diagnostics:** Every score is fully traceable to measurable physical angles, normalized stride ratios, and torso alignment metrics rather than opaque deep-learning classification scores.

---

# 5. Objectives

### 5.1 Implemented Objectives (Active in Codebase)
- [x] **Monocular Video Ingestion:** Accept standard consumer video formats (`.mp4`, `.avi`, `.mov`) without specialized hardware.
- [x] **Lighting Normalization:** Implement CLAHE lightness equalization in the LAB color space to mitigate shadows and overexposure.
- [x] **15-Joint Skeletal Tracking:** Extract 15 standardized anatomical keypoints via Google MediaPipe Pose with continuous visibility confidence filtering.
- [x] **Automated Handedness Detection:** Determine batter stance (Right-Handed vs. Left-Handed) from temporal wrist extension dynamics.
- [x] **Multi-Shot Phase Detection:** Dynamically segment stroke sequences into distinct temporal phases (Stance, Downswing/Preparation, Contact Window, Follow-Through) for Cover Drive, Straight Drive, and Pull Shot.
- [x] **12-Dimensional Kinematic Feature Extraction:** Extract scale-invariant metrics (joint angles, stride ratios, alignment offsets, and spatial displacement).
- [x] **Data-Driven Shot Classification:** Classify batting strokes into Cover Drive, Straight Drive, Pull Shot, or Unknown with top-class margin gating and confidence thresholds.
- [x] **Multi-Signal Bat–Ball Contact Estimation:** Combine bat blade segmentation and ball tracking to detect visual contact, calculate bat-ball proximity, estimate deflection angles, and separate visual contact from pose impact.
- [x] **Empirical Multi-Shot Calibration ($N=30$):** Maintain dedicated reference datasets ($N=10$ per class) and calculate 14 statistical descriptors (Mean, Median, Std, Var, Min, Max, $P_{10}, P_{25}, P_{50}, P_{75}, P_{90}$, IQR, and Outlier Bounds) exported to structured JSON and CSV databases.
- [x] **Continuous Distribution Scoring ($0-100$):** Score execution using non-linear percentile curves supporting two-sided optimal, lower-is-better, and higher-is-better policies.
- [x] **Dynamic Occlusion Renormalization:** Renormalize metric and category weights dynamically when limbs are occluded, preventing unfair zero-score penalties.
- [x] **Streamlit Multi-Tab Web Application:** Deliver an interactive 3-tab dashboard for single video analysis, multi-shot batch calibration, and human-in-the-loop contact annotation.
- [x] **Automated Architectural & Multi-Shot Unit Testing:** Maintain a 35-case automated test suite verifying all pipeline components without regression.

### 5.2 Planned Future Objectives (Roadmap / Not Implemented)
- [ ] **Fast Bowling Delivery Phasing:** Front-foot contact, back-foot contact, release window, and follow-through analysis for bowlers.
- [ ] **Deep Learning Pose Ensembles:** Integration of RTMPose-M or MMPose for higher keypoint resolution under extreme motion blur.
- [ ] **Spatio-Temporal Graph Neural Networks (ST-GCN / CTR-GCN):** Deep skeleton action recognition directly from joint coordinate graphs.
- [ ] **High-Dimensional Vector Retrieval:** FAISS vector similarity search across thousands of professional stroke embeddings.
- [ ] **On-Device Real-Time Edge Processing:** Sub-30ms mobile inference pipeline for live net sessions.

---

# 6. Project Scope

The current implementation of AthleteRise is strictly focused on **Cricket Batting Kinematics**.

| Dimension | Supported in Active Implementation | Not Supported / Out of Scope |
| :--- | :--- | :--- |
| **Cricket Discipline** | Batting stroke mechanics | Bowling, wicketkeeping, and fielding |
| **Supported Strokes** | Cover Drive, Straight Drive, Pull Shot | Cut, Hook, Sweep, Glance, Reverse Sweep |
| **Camera Configuration**| Single stationary monocular camera (side-on or 45° angle) | Multi-camera stereoscopic 3D setups |
| **Video Input** | Recorded files (`.mp4`, `.avi`, `.mov`) | Low-latency live broadcast RTSP streams |
| **Pose Estimation** | MediaPipe Tasks & Legacy Solutions (BlazePose) | RTMPose-M, AlphaPose, OpenPose |
| **Reference Baselines** | Empirical $N=10$ per shot class (total $N=30$) | Multi-thousand video international professional database |
| **Classification** | Standardized Euclidean prototype distance + Softmax | Deep neural networks, SVMs, or Random Forests |

---

# 7. System Overview

```
                      +-----------------------------+
                      |   Monocular Cricket Video   |
                      |      (.mp4, .avi, .mov)     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     VideoPreprocessor       |
                      |   - OpenCV VideoCapture     |
                      |   - CLAHE LAB Normalization |
                      |   - Frame Extraction        |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    VisionPostureTracker     |
                      |   - MediaPipe Tasks API     |
                      |   - 15 Standardized Joints  |
                      |   - Optical Flow Fallback   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Handedness Classifier   |
                      |  - Wrist Extension Dynamics |
                      |  - Stance: Right or Left    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Kinematic Feature Engine |
                      | - 12D Spatio-Temporal Vector|
                      | - Preliminary Impact Window |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |        ShotClassifier       |
                      |  - Standardized Prototype D |
                      |  - Softmax & Margin Gating  |
                      |  - Class: CD / SD / PS / UNK|
                      +--------------+--------------+
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
    +---------------------------+         +---------------------------+
    |    PhaseDetectorFactory   |         |    Bat & Ball Trackers    |
    | - Shot-Specific Phasing   |         | - BatDetector (HSV/Aspect)|
    | - Pose Impact Frame (tp)  |         | - BallDetector (HSV/Diff) |
    | - Stance, Swing, FT       |         | - Trajectory Continuity   |
    +-------------+-------------+         +-------------+-------------+
                  |                                     |
                  +------------------+------------------+
                                     |
                                     v
                      +-----------------------------+
                      |       ContactEstimator      |
                      | - Multi-Signal Evidence     |
                      | - Spatial Proximity         |
                      | - Ball Deflection Angle     |
                      | - Visual Contact Frame (tc) |
                      | - Status: Contact/No/Unknown|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   MultiShotScoringEngine    |
                      | - Select Calibrated Baseline|
                      | - 10 Core Biomechanical Ms  |
                      | - Shot-Specific Indicators  |
                      | - Continuous P10-P90 Curves |
                      | - Dynamic NA Renormalization|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Reporting & UI Delivery   |
                      | - output/annotated_video.mp4|
                      | - output/evaluation.json    |
                      | - output/evaluation_report  |
                      | - Streamlit 3-Tab Dashboard |
                      +-----------------------------+
```

---

# 8. Complete Architecture

### 8.1 Architectural Layers

1. **Ingestion & Preprocessing Layer (`src/preprocessing/`):**
   Decodes video streams, extracts intrinsic frame dimensions and framerates, handles optional region-of-interest cropping, and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the Lightness channel of the LAB color space.

2. **Computer Vision & Pose Tracking Layer (`src/pose_estimation/`):**
   Wraps the MediaPipe Tasks PoseLandmarker (`pose_landmarker_lite.task`) and legacy `mp.solutions.pose` engine. Tracks 15 key anatomical joints with visibility confidence filtering. Incorporates frame-differencing optical flow and silhouette propagation to prevent landmark dropout.

3. **Kinematic Analysis & Classification Layer (`src/shot_analysis/`, `src/biomechanics/`):**
   Computes frame-level joint angles and alignments. Automatically classifies batter handedness. Extracts a 12-dimensional spatio-temporal feature vector. Evaluates standardized Euclidean distance to class prototypes with pooled variance normalization, outputting class probabilities, margins, and confidence levels.

4. **Phase Detection Layer (`src/phase_detection/`):**
   Dispatches specialized phase detectors via `PhaseDetectorFactory`:
   - `CoverDrivePhaseDetector`: Identifies contact moment $t_p$ via front-elbow inflection during downswing followed by follow-through extension.
   - `StraightDrivePhaseDetector`: Identifies $t_p$ via minimal head-over-knee horizontal offset and downward wrist progression.
   - `PullShotPhaseDetector`: Identifies $t_p$ via peak shoulder rotational uncoiling velocity.

5. **Contact Detection Layer (`src/contact_detection/`):**
   Runs modular `BatDetector` (HSV willow wood segmentation, wrist ROI filtering, aspect-ratio contour fitting) and `BallDetector` (dual-range HSV red ball segmentation, frame differencing, circularity check). Combines signals in `ContactEstimator` to output a unified 3-state decision (`CONTACT_DETECTED`, `NO_CONTACT_DETECTED`, `UNKNOWN`) and visual contact frame $t_c$.

6. **Calibration & Scoring Layer (`src/calibration/`, `src/evaluation/`):**
   Maintains independent statistical baselines for Cover Drive, Straight Drive, and Pull Shot ($N=10$ per class). Evaluates user contact-window medians against continuous percentile distributions ($P_{10} - P_{90}$). Aggregates metric scores into five functional categories with dynamic occlusion weight renormalization.

7. **Reporting & Presentation Layer (`src/reporting/`, `src/report/`, `app.py`):**
   Renders annotated video with skeletal overlays, bat/ball trajectory trails, and decision banners. Generates JSON and structured text reports. Serves the interactive 3-tab Streamlit dashboard.

---

# 9. Repository Structure

The current codebase is organized as follows:

```
AI-Powered-Cricket-Analytics-main/
│
├── README.md                                # Project overview, quick start, and navigation
├── app.py                                   # Streamlit Web Application (3-Tab Dashboard)
├── calibrate_references.py                  # Batch reference calibration orchestrator
├── config.json                              # Fallback expert heuristic rules and pipeline configs
├── cover_drive_analysis_realtime.py         # Main standalone video analysis pipeline
├── pose_landmarker_lite.task                # MediaPipe Tasks PoseLandmarker model binary
├── requirements.txt                         # Python dependencies
│
├── src/                                     # Core Source Code Directory
│   ├── biomechanics/                        # Biomechanical calculations
│   │   ├── metrics.py                       # 10 core batting metrics, handedness, angles
│   │   └── shot_specific_metrics.py         # Specialized metrics for Cover, Straight, and Pull
│   ├── calibration/                         # Reference calibration engine
│   │   ├── base_calibration_engine.py       # Base calibration engine & statistical calculations
│   │   ├── calibration_engine.py            # Legacy cover drive calibration engine
│   │   ├── calibration_factory.py           # Factory dispatching shot calibration engines
│   │   ├── cover_drive_calibration.py       # Cover drive calibration wrapper
│   │   ├── pull_shot_calibration.py         # Pull shot calibration wrapper
│   │   └── straight_drive_calibration.py    # Straight drive calibration wrapper
│   ├── contact_detection/                   # Bat-ball tracking & contact estimation
│   │   ├── annotation_manager.py            # Ground truth annotation manager (Tab 3)
│   │   ├── bat_ball_detector.py             # Modular BatDetector and BallDetector
│   │   ├── contact_estimator.py             # Multi-signal contact estimator engine
│   │   ├── contact_features.py              # Geometric point-to-segment distance & deflection
│   │   └── contact_validator.py             # Validation utilities for contact evaluation
│   ├── evaluation/                          # Scoring and evaluation engines
│   │   ├── base_scoring_engine.py           # MultiShotScoringEngine & continuous scoring curves
│   │   ├── cover_drive_scoring.py           # Cover drive scoring engine wrapper
│   │   ├── feedback_engine.py               # Template-based explainable coaching feedback
│   │   ├── pull_shot_scoring.py             # Pull shot scoring engine wrapper
│   │   ├── scoring_engine.py                # Hierarchical category weighting & expert fallback
│   │   └── straight_drive_scoring.py        # Straight drive scoring engine wrapper
│   ├── phase_detection/                     # Temporal phase segmentation
│   │   ├── base_phase_detector.py           # Abstract base class for phase detection
│   │   ├── cover_drive_phase_detector.py    # Elbow inflection & downswing detector
│   │   ├── phase_detector.py                # Legacy phase detector wrapper
│   │   ├── phase_factory.py                 # Dynamic phase detector factory
│   │   ├── pull_shot_phase_detector.py      # Shoulder uncoiling rotation detector
│   │   └── straight_drive_phase_detector.py # Head-over-knee vertical alignment detector
│   ├── pose_estimation/                     # Skeletal keypoint extraction
│   │   └── posture_tracker.py               # VisionPostureTracker (Tasks API + flow fallback)
│   ├── preprocessing/                       # Video ingestion and lighting
│   │   └── video_preprocessing.py           # VideoPreprocessor with CLAHE LAB normalization
│   ├── report/                              # Calibration reporting
│   │   └── calibration_reporter.py          # Calibration reporter, HTML generation, PNG plots
│   ├── reporting/                           # Evaluation reporting
│   │   ├── contact_report.py                # Contact analysis text formatter
│   │   └── multi_shot_report.py             # MultiShotReportGenerator for evaluation output
│   └── shot_analysis/                       # Multi-shot classification
│       ├── shot_classifier.py               # Prototype distance classifier with LOO-CV
│       ├── shot_config.py                   # Registry metadata, shot codes, directory mappings
│       ├── shot_features.py                 # 12D kinematic feature extractor with quality tracking
│       └── shot_registry.py                 # Shot metadata registry
│
├── data/                                    # Datasets & Reference Media
│   ├── contact_annotations/                 # Ground truth contact annotations from Tab 3
│   └── reference_videos/                    # Empirical reference video datasets (N=30)
│       ├── cover_drive/                     # 10 proficient Cover Drive reference clips
│       ├── pull_shot/                       # 10 proficient Pull Shot reference clips
│       └── straight_drive/                  # 10 proficient Straight Drive reference clips
│
├── calibration_dataset/                     # Secondary / user-provided calibration directory
│   └── cover_drive/                         # User reference video directory
│
├── examples/                                # Reference demonstration GIF/PNG assets
│
├── output/                                  # Generated Analysis & Calibration Artifacts
│   ├── annotated_video.mp4                  # Visual annotated video with HUD overlays
│   ├── classification_debug.json            # Full feature vector, probabilities, and margins
│   ├── evaluation.json                      # Complete structured analysis JSON payload
│   ├── evaluation_report.txt                # Comprehensive formatted text evaluation report
│   ├── temporal_smoothness.png              # Matplotlib kinematics curves across frames
│   └── reference_calibration/               # Exported calibration baselines
│       ├── calibration_diagnostics.txt      # Cross-shot comparative diagnostic matrix
│       ├── cover_drive/                     # cover_drive_reference.json, CSVs, HTML report
│       ├── pull_shot/                       # pull_shot_reference.json, CSVs
│       └── straight_drive/                  # straight_drive_reference.json, CSVs
│
├── scripts/                                 # Diagnostic & Evaluation Scripts
│   ├── check_aligned_tilt.py                # Spine tilt verification
│   ├── check_cover_shoulders.py             # Cover drive shoulder angle audit
│   ├── check_facing.py                      # Batter facing orientation check
│   ├── check_input_video_stride.py          # Video stride length calculation check
│   ├── check_shoulders.py                   # Shoulder rotation validation
│   ├── compare_cover_straight.py            # Cross-shot comparison script
│   ├── evaluate_dataset.py                  # Full 30-sample re-substitution evaluation
│   ├── evaluate_models.py                   # Experimental scikit-learn model benchmark
│   ├── generate_multishot_dataset.py        # Dataset generation helper
│   ├── inspect_per_video.py                 # Per-video metrics inspection
│   ├── inspect_stats.py                     # Statistical summary inspector
│   └── test_weighted_classifier.py          # Prototype classifier unit test script
│
└── tests/                                   # Automated Unit Test Suites
    ├── test_audit_architecture.py           # 7 forensic audit tests (continuous scoring, N/A)
    ├── test_audit_step19_pipeline.py        # 11 end-to-end multi-shot pipeline scenarios
    └── test_multi_shot_analysis.py          # 17 multi-shot classification, calibration, & contact tests
```

---

# 10. Technology Stack

| Technology / Library | Version / Requirement | Role in System | Actual Verified Usage |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10.x | Runtime execution environment | Core execution, type annotations, math operations. |
| **OpenCV (`opencv-python`)** | $\ge 4.8.0$ | Computer Vision & Video I/O | Frame decoding, CLAHE in LAB space, HSV color segmentation, contour analysis, video writer. |
| **Google MediaPipe (`mediapipe`)** | $\ge 0.10.0$ | Markerless Pose Estimation | BlazePose 33-landmark estimation; uses `pose_landmarker_lite.task` via Tasks Vision API and legacy `solutions.pose`. |
| **NumPy (`numpy`)** | $\ge 1.23.0$ | Numerical Computation | Vector trigonometry, 2D interior joint angles, percentiles ($P_{10} - P_{90}$), variance, and Euclidean distances. |
| **Matplotlib (`matplotlib`)** | $\ge 3.7.0$ | Data Visualization & Plotting | Kinematic temporal smoothness curves (`output/temporal_smoothness.png`), radar charts, correlation heatmaps. |
| **Streamlit (`streamlit`)** | $\ge 1.28.0$ | Interactive Web User Interface | 3-tab web dashboard: video upload, real-time telemetry display, batch calibration, contact review. |
| **Scikit-learn (`scikit-learn`)**| Standard | Baseline Model Benchmarking | Leave-One-Out cross-validation benchmarks in `scripts/evaluate_models.py`. (Core classifier is native NumPy). |
| **Python Standard `unittest`** | Built-in | Verification & Test Suite | Automated execution of 35 test cases across 3 test modules in `tests/`. |

---

# 11. Complete Data Flow

```
Raw Monocular Video (.mp4 / .avi)
  │
  ▼ [VideoPreprocessor]
Processed BGR Frames (CLAHE LAB normalized)
  │
  ▼ [VisionPostureTracker]
15 Keypoint Coordinate Dict: {joint: [x_norm, y_norm, visibility]}
  │
  ├──────────────────────────────┐
  ▼ [detect_handedness]           ▼ [BatDetector & BallDetector]
Stance: 'right' | 'left'       Bat BBox/Segment & Ball Trajectory
  │                              │
  ▼ [ShotFeatureExtractor]       │
12D Kinematic Feature Vector     │
  │                              │
  ▼ [ShotClassifier]             │
Predicted Class + Margin + Prob  │
  │                              │
  ▼ [PhaseDetectorFactory]       │
Pose Impact Frame (tp) + Phases  │
  │                              │
  └──────────────┬───────────────┘
                 │
                 ▼ [ContactEstimator]
Visual Bat-Ball Contact Frame (tc) + Status (CONTACT / NO_CONTACT / UNKNOWN)
                 │
                 ▼ [MultiShotScoringEngine]
Selected Reference Baseline (cover_drive / straight_drive / pull_shot)
                 │
                 ├─► 10 Core Biomechanical Metrics (Median at Contact Window)
                 ├─► Shot-Specific Kinematic Indicators
                 ├─► Continuous Metric Scores (0-100) via P10-P90 Curves
                 ├─► 5 Category Scores with Dynamic N/A Renormalization
                 ├─► Overall Technique Score (0-100) & Skill Grade
                 └─► Template-Driven Coaching Feedback
                 │
                 ▼
Outputs:
  - output/annotated_video.mp4 (HUD Overlays + Trajectory Trails)
  - output/evaluation.json (Full Structured Telemetry)
  - output/evaluation_report.txt (Text Summary)
  - Streamlit UI Dashboard
```

---

# 12. Video Processing Pipeline

The video processing pipeline is implemented in [`src/preprocessing/video_preprocessing.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/preprocessing/video_preprocessing.py) through the `VideoPreprocessor` class.

### 12.1 Video Ingestion
The video stream is loaded via `cv2.VideoCapture`. The preprocessor queries:
- `cv2.CAP_PROP_FRAME_WIDTH`: Native frame width $W$.
- `cv2.CAP_PROP_FRAME_HEIGHT`: Native frame height $H$.
- `cv2.CAP_PROP_FPS`: Native frame rate (defaults to 30 if metadata is corrupted or missing).
- `cv2.CAP_PROP_FRAME_COUNT`: Total frame count $T$.

### 12.2 Lighting Normalization via CLAHE
To handle outdoor cricket ground lighting variations (harsh midday sunlight, deep pitch shadows, artificial floodlights), the system applies Contrast Limited Adaptive Histogram Equalization:
1. The BGR frame is converted to the **CIE LAB color space**, which separates Lightness ($L^*$) from chromaticity ($a^*, b^*$).
2. CLAHE is applied exclusively to the $L^*$ channel with a **clip limit of 2.0** and a **tile grid size of $(8 \times 8)$**. This enhances local contrast without amplifying high-frequency noise.
3. The equalized $L^*$ channel is merged with the original $a^*$ and $b^*$ channels and converted back to BGR.

### 12.3 Frame Extraction Generator
Frames are processed sequentially via Python generator `read_frames()`, yielding `(frame_idx, processed_frame)`. If configured, extracted frames are cached to `frames/frame_000001.jpg`.

---

# 13. Pose Estimation

Pose estimation is implemented in [`src/pose_estimation/posture_tracker.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/pose_estimation/posture_tracker.py) by `VisionPostureTracker`.

### 13.1 Multi-Tier Tracking Hierarchy
To ensure continuous tracking across diverse hardware and operating systems, `VisionPostureTracker` implements a multi-tier fallback architecture:
1. **MediaPipe Tasks Vision API:** Ingests `pose_landmarker_lite.task` via `mediapipe.tasks.python.vision.PoseLandmarker`.
2. **MediaPipe Legacy Solutions API:** Ingests `mediapipe.solutions.pose.Pose` (`min_detection_confidence=0.4, min_tracking_confidence=0.4`).
3. **Temporal Optical-Flow Propagation:** If MediaPipe drops detection on a frame (e.g., due to extreme motion blur during downswing), the tracker computes the displacement vector $(\Delta x, \Delta y)$ using absolute frame differencing between consecutive grayscale frames and propagates the last valid landmarks with a confidence decay factor ($0.96$).
4. **Silhouette Bounding Box Fallback:** If tracking fails before initialization, the system uses binary threshold contouring to approximate body torso position and project initial joint anchors.

### 13.2 15 Standardized Anatomical Keypoints
The raw 33 MediaPipe landmarks are filtered down to 15 key joints essential for batting kinematics:

| Standard Key | MediaPipe Landmark ID | Description |
| :--- | :---: | :--- |
| `nose` | 0 | Head apex and visual orientation anchor |
| `left_shoulder` | 11 | Lead shoulder (RHB) / Rear shoulder (LHB) |
| `right_shoulder`| 12 | Rear shoulder (RHB) / Lead shoulder (LHB) |
| `left_elbow` | 13 | Lead elbow (RHB) / Rear elbow (LHB) |
| `right_elbow` | 14 | Rear elbow (RHB) / Lead elbow (LHB) |
| `left_wrist` | 15 | Lead wrist (RHB) / Rear wrist (LHB) |
| `right_wrist` | 16 | Rear wrist (RHB) / Lead wrist (LHB) |
| `left_hip` | 23 | Lead hip (RHB) / Rear hip (LHB) |
| `right_hip` | 24 | Rear hip (RHB) / Lead hip (LHB) |
| `left_knee` | 25 | Lead knee (RHB) / Rear knee (LHB) |
| `right_knee` | 26 | Rear knee (RHB) / Lead knee (LHB) |
| `left_ankle` | 27 | Lead ankle (RHB) / Rear ankle (LHB) |
| `right_ankle` | 28 | Rear ankle (RHB) / Lead ankle (LHB) |
| `left_foot_index` | 31 | Lead toe direction anchor (RHB) |
| `right_foot_index`| 32 | Rear toe direction anchor (RHB) |

Coordinates are represented as normalized floating-point triplets $[x, y, \text{visibility}]$ where $x, y \in [0.0, 1.0]$.

### 13.3 Automated Handedness / Stance Classification
Implemented in [`src/biomechanics/metrics.py:detect_handedness`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/biomechanics/metrics.py#L20-L52):
$$\Delta x_{\text{left}} = \frac{1}{N} \sum_{t=1}^N |x_{\text{left\_wrist}}(t) - x_{\text{left\_shoulder}}(t)|$$
$$\Delta x_{\text{right}} = \frac{1}{N} \sum_{t=1}^N |x_{\text{right\_wrist}}(t) - x_{\text{right\_shoulder}}(t)|$$

$$\text{Stance} = \begin{cases} \text{'right'} & \text{if } \Delta x_{\text{left}} \ge 0.85 \cdot \Delta x_{\text{right}} \\ \text{'left'} & \text{otherwise} \end{cases}$$

This dynamically assigns the **lead** vs **rear** anatomical limbs.

---

# 14. Biomechanical Analysis

Implemented in [`src/biomechanics/metrics.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/biomechanics/metrics.py) and [`src/biomechanics/shot_specific_metrics.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/biomechanics/shot_specific_metrics.py).

### 14.1 Core Biomechanical Metrics

#### 1. Front Elbow Angle (`front_elbow_angle`)
- **Definition:** 2D interior joint angle at the lead elbow formed by lead shoulder, elbow, and wrist.
- **Formula:**
  $$\theta = \left| \operatorname{atan2}(w_y - e_y, w_x - e_x) - \operatorname{atan2}(s_y - e_y, s_x - e_x) \right| \times \frac{180^\circ}{\pi}$$
  $$\text{front\_elbow\_angle} = \begin{cases} 360^\circ - \theta & \text{if } \theta > 180^\circ \\ \theta & \text{otherwise} \end{cases}$$
- **Required Landmarks:** `front_shoulder`, `front_elbow`, `front_wrist`
- **Validity Gating:** Visibility $\ge 0.35$; anatomically constrained to $[25^\circ, 180^\circ]$.
- **Physical Interpretation:** Measures front-arm elevation and vertical bat leverage. Higher angle represents a dominant high leading elbow.

#### 2. Spine Lean Tilt (`spine_lean`)
- **Definition:** Deviation of the lead torso axis from the true vertical vector.
- **Formula:**
  $$\theta_{\text{raw}} = \operatorname{calculate\_angle}(\text{front\_hip}, \text{front\_shoulder}, [s_x, s_y - 1.0])$$
  $$\text{spine\_lean} = |180^\circ - \theta_{\text{raw}}|$$
- **Required Landmarks:** `front_hip`, `front_shoulder`
- **Validity Gating:** Visibility $\ge 0.5$; anatomical integrity check requires $s_y < h_y$ (shoulder above hip in 2D image coordinates). Filtered to $\le 60^\circ$.
- **Physical Interpretation:** Quantifies lateral torso lean and core balance over the center of gravity.

#### 3. Head-Over-Knee Vertical Alignment (`head_knee_alignment`)
- **Definition:** Horizontal distance between nose and lead knee, normalized by torso length.
- **Formula:**
  $$\text{torso\_length} = \|\mathbf{p}_{\text{mid\_shoulder}} - \mathbf{p}_{\text{mid\_hip}}\|_2$$
  $$\text{head\_knee\_alignment} = \frac{|x_{\text{nose}} - x_{\text{front\_knee}}|}{\text{torso\_length} + \epsilon}$$
- **Required Landmarks:** `nose`, `front_knee`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`
- **Physical Interpretation:** Scale-invariant measure of head positioning over the front knee. Lower values indicate ideal vertical head-over-ball positioning.

#### 4. Front Foot Direction (`front_foot_direction`)
- **Definition:** 2D interior angle formed by the lead knee, lead ankle, and lead foot index (toe).
- **Formula:** $\operatorname{calculate\_angle}(\text{front\_knee}, \text{front\_ankle}, \text{front\_foot\_index})$
- **Required Landmarks:** `front_knee`, `front_ankle`, `front_foot_index`
- **Validity Gating:** Minimum visibility $\ge 0.40$; returns `None` if occluded.
- **Physical Interpretation:** Quantifies front-foot alignment toward target stroke zones.

#### 5. Shoulder Rotation (`shoulder_rotation`)
- **Definition:** Absolute azimuth angle of the vector connecting left and right shoulders.
- **Formula:**
  $$\theta_{\text{sh}} = |\operatorname{atan2}(s_{R, y} - s_{L, y}, s_{R, x} - s_{L, x})| \times \frac{180^\circ}{\pi}$$
  $$\text{shoulder\_rotation} = \begin{cases} 180^\circ - \theta_{\text{sh}} & \text{if } \theta_{\text{sh}} < 90^\circ \\ \theta_{\text{sh}} & \text{otherwise} \end{cases}$$
- **Required Landmarks:** `left_shoulder`, `right_shoulder`
- **Physical Interpretation:** Quantifies shoulder uncoiling invariance across camera angles.

#### 6. Hip Rotation (`hip_rotation`)
- **Definition:** Absolute azimuth angle of the vector connecting left and right hips.
- **Formula:**
  $$\theta_{\text{hip}} = |\operatorname{atan2}(h_{R, y} - h_{L, y}, h_{R, x} - h_{L, x})| \times \frac{180^\circ}{\pi}$$
  $$\text{hip\_rotation} = \begin{cases} 180^\circ - \theta_{\text{hip}} & \text{if } \theta_{\text{hip}} < 90^\circ \\ \theta_{\text{hip}} & \text{otherwise} \end{cases}$$
- **Required Landmarks:** `left_hip`, `right_hip`
- **Physical Interpretation:** Measures pelvic opening and kinetic chain transfer through contact.

#### 7. Stride Length (`stride_length`)
- **Definition:** Inter-ankle Euclidean distance normalized by lead leg length.
- **Formula:**
  $$\text{stride\_length} = \frac{\|\mathbf{p}_{\text{front\_ankle}} - \mathbf{p}_{\text{back\_ankle}}\|_2}{\|\mathbf{p}_{\text{front\_hip}} - \mathbf{p}_{\text{front\_ankle}}\|_2 + \epsilon}$$
- **Required Landmarks:** `front_ankle`, `back_ankle`, `front_hip`
- **Physical Interpretation:** Body-scale normalized base of support.

#### 8. Head Stability (`head_stability`)
- **Definition:** Combined variance of normalized nose coordinates during the downswing phase.
- **Formula:**
  $$x_{\text{norm}}(t) = \frac{x_{\text{nose}}(t)}{w_{\text{shoulder}}}, \quad y_{\text{norm}}(t) = \frac{y_{\text{nose}}(t)}{w_{\text{shoulder}}}$$
  $$\text{head\_stability} = \sqrt{\operatorname{Var}(x_{\text{norm}}) + \operatorname{Var}(y_{\text{norm}})}$$
- **Required Landmarks:** `nose`, `left_shoulder`, `right_shoulder` across $t \in [t_{\text{downswing}}, t_{\text{contact}}]$.
- **Physical Interpretation:** Quantifies head movement during the downswing; lower values indicate a still, steady head.

#### 9. Follow-Through Angle (`follow_through_angle`)
- **Definition:** Mean front elbow extension angle during the follow-through window ($t_c + 4$ to $t_c + 20$).
- **Formula:** $\frac{1}{M} \sum_{t=t_c+4}^{t_{\text{end}}} \theta_{\text{front\_elbow}}(t)$
- **Required Landmarks:** `front_shoulder`, `front_elbow`, `front_wrist`
- **Physical Interpretation:** Evaluates whether the stroke was completed with full fluid arm extension.

#### 10. Follow-Through Quality (`follow_through_quality`)
- **Definition:** Normalized percentage of full follow-through completion (scaled against a reference $150^\circ$ extension).
- **Formula:** $\operatorname{clip}\left(\frac{\text{follow\_through\_angle}}{150.0^\circ} \times 100.0, 0.0, 100.0\right)$

---

### 14.2 Shot-Specific Kinematic Metrics

Implemented in [`src/biomechanics/shot_specific_metrics.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/biomechanics/shot_specific_metrics.py):

#### Cover Drive Specifics
- `cover_direction_alignment`: Front foot direction angle toward cover zone ($140^\circ - 175^\circ$).
- `bat_extension_index`: Normalized front elbow reach $\operatorname{clip}(\theta_{\text{elbow}} / 180^\circ, 0.0, 1.0)$.
- `offside_flow_vector`: Combined stride and shoulder uncoiling vector: $0.5 \cdot \theta_{\text{shoulder}} + 0.5 \cdot (\text{stride} \times 100)$.

#### Straight Drive Specifics
- `straight_bat_presentation`: Front foot alignment directed straight down the pitch ($85^\circ - 115^\circ$).
- `head_over_ball_verticality`: Inverted head-knee offset index: $\max(0.0, 1.0 - \text{head\_knee\_alignment})$.
- `leading_elbow_elevation`: Front elbow angle at contact.

#### Pull Shot Specifics
- `rotary_trunk_uncoiling`: Shoulder rotation angle uncoiling through mid-wicket / square leg.
- `horizontal_plane_index`: Cross-bat indicator: $\operatorname{clip}((\text{spine\_lean} / 30^\circ) \times 50 + (\theta_{\text{elbow}} / 150^\circ) \times 50, 0.0, 100.0)$.
- `rear_foot_pivot_compactness`: Back-foot compactness index: $\max(0.0, 1.0 - \text{stride\_length})$.

---

# 15. Cover Drive Analysis

The Cover Drive is a front-foot attacking stroke executed through the off-side (extra cover region). 

### 15.1 Biomechanical Requirements Verified in Code
1. **Front Foot Placement:** Front foot strides forward towards the pitch of the ball, opening at an angle between $140^\circ$ and $175^\circ$ toward the cover boundary to allow the hips and bat to swing through freely.
2. **Head Alignment:** Head is positioned directly over the lead knee at contact ($\text{head\_knee\_alignment} \le 0.25$ normalized offset), ensuring the center of gravity transfers through the stroke.
3. **Leading Elbow Elevation:** High front elbow ($\ge 120^\circ$) leading the downswing, establishing a vertical bat face and preventing slicing.
4. **Upright Posture:** Minimal lateral torso lean ($\text{spine\_lean} \le 20^\circ$) maintaining rotational balance over the base of support.
5. **High Follow-Through:** Full extension through contact, completing with front elbow elevated ($\ge 130^\circ$).

### 15.2 Phase Segmentation Algorithm
Implemented in `CoverDrivePhaseDetector`:
- Identifies candidates where front elbow reaches local minimum followed by subsequent follow-through extension of $\ge 25^\circ$:
  $$t_p = \arg\min_{t \in [0.2T, 0.85T]} \left( \theta_{\text{front\_elbow}}(t) - 0.35 \cdot \text{rise}(t) \right)$$
- Temporal phases:
  - **Stance:** $[1, \max(1, t_p - 16)]$
  - **Downswing:** $[\max(1, t_p - 15), t_p - 1]$
  - **Contact Window:** $[\max(1, t_p - 3), \min(T, t_p + 3)]$
  - **Follow-Through:** $[\min(T, t_p + 4), \min(T, t_p + 20)]$

---

# 16. Reference Dataset and Calibration

### 16.1 Reference Dataset Inventory
AthleteRise includes 30 verified empirical reference videos stored in `data/reference_videos/`:
- `data/reference_videos/cover_drive/`: 10 videos (`cover_0001.avi` to `cover_0010.avi`)
- `data/reference_videos/straight_drive/`: 10 videos (`straight_0001.avi` to `straight_0010.avi`)
- `data/reference_videos/pull_shot/`: 10 videos (`pull_0001.avi` to `pull_0010.avi`)

### 16.2 Calibration Engine Execution
Implemented in [`src/calibration/base_calibration_engine.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/calibration/base_calibration_engine.py). For each shot class, `ShotCalibrationEngine`:
1. Ingests all reference clips in the target directory.
2. Extracts frame landmarks, handedness, impact frames, and contact-window medians.
3. Computes 14 statistical descriptors across all valid samples:
   - **Central Tendency:** Mean ($\mu$), Median ($P_{50}$)
   - **Dispersion:** Standard Deviation ($\sigma$), Variance ($\sigma^2$), Interquartile Range ($\text{IQR} = P_{75} - P_{25}$)
   - **Extremes:** Minimum, Maximum
   - **Percentile Curves:** $P_{10}, P_{25}, P_{50}, P_{75}, P_{90}$
   - **Outlier Fences:** $[\text{Lower} = P_{25} - 1.5 \cdot \text{IQR}, \text{Upper} = P_{75} + 1.5 \cdot \text{IQR}]$
   - **Diagnostics:** Identifies any reference video exceeding outlier fences.
4. Exports structured JSON baselines:
   - `output/reference_calibration/cover_drive/cover_drive_reference.json`
   - `output/reference_calibration/straight_drive/straight_drive_reference.json`
   - `output/reference_calibration/pull_shot/pull_shot_reference.json`
5. Exports CSV metric tables (`*_statistics.csv`, `*_video_metrics.csv`) and generates cross-shot diagnostics (`calibration_diagnostics.txt`).

### 16.3 Actual Measured Baseline Statistics

*Empirical reference matrix extracted from `output/reference_calibration/calibration_diagnostics.txt`:*

| Metric | Cover Mean ($\mu$) | Cover Std ($\sigma$) | Straight Mean ($\mu$) | Straight Std ($\sigma$) | Pull Mean ($\mu$) | Pull Std ($\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Front Elbow Angle (°)** | $120.45^\circ$ | $44.18^\circ$ | $140.46^\circ$ | $16.45^\circ$ | $126.33^\circ$ | $48.94^\circ$ |
| **Spine Lean Tilt (°)** | $18.25^\circ$ | $9.20^\circ$ | $10.70^\circ$ | $5.93^\circ$ | $11.25^\circ$ | $8.22^\circ$ |
| **Head-Knee Alignment (norm)** | $0.25$ | $0.17$ | $0.11$ | $0.07$ | $0.14$ | $0.09$ |
| **Front Foot Direction (°)** | $156.21^\circ$ | $11.57^\circ$ | $167.70^\circ$ | $7.28^\circ$ | $158.06^\circ$ | $16.46^\circ$ |
| **Shoulder Rotation (°)** | $143.95^\circ$ | $21.90^\circ$ | $150.26^\circ$ | $16.91^\circ$ | $168.98^\circ$ | $7.66^\circ$ |
| **Hip Rotation (°)** | $144.38^\circ$ | $18.85^\circ$ | $160.20^\circ$ | $15.44^\circ$ | $155.22^\circ$ | $19.30^\circ$ |
| **Stride Length (norm)** | $0.55$ | $0.22$ | $0.67$ | $0.29$ | $0.58$ | $0.24$ |
| **Head Stability (norm)** | $24.63$ | $24.83$ | $65.02$ | $86.82$ | $28.39$ | $12.64$ |
| **Follow-Through Angle (°)** | $132.58^\circ$ | $44.65^\circ$ | $130.27^\circ$ | $34.21^\circ$ | $135.02^\circ$ | $28.71^\circ$ |
| **Follow-Through Quality (0-100)**| $83.45$ | $26.17$ | $82.73$ | $18.44$ | $86.86$ | $15.69$ |

---

# 17. Statistical / Threshold System

AthleteRise implements a continuous distribution scoring system rather than binary threshold gates.

### 17.1 Continuous Scoring Policies (`score_metric_continuous`)

Implemented in [`src/evaluation/base_scoring_engine.py:L120-L219`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/evaluation/base_scoring_engine.py#L120-L219):

#### Policy 1: Two-Sided Optimal (e.g., `front_foot_direction`, `stride_length`)
Evaluates metrics where both extreme undershoot and extreme overshoot are technically flawed:
- **Core Reference Range ($P_{25} \le x \le P_{75}$):** Score is scaled continuously between $90.0$ and $100.0$:
  $$\text{Score} = \begin{cases} 90.0 + 10.0 \cdot \frac{x - P_{25}}{P_{50} - P_{25}} & \text{if } x < P_{50} \\ 100.0 - 10.0 \cdot \frac{x - P_{50}}{P_{75} - P_{50}} & \text{if } x \ge P_{50} \end{cases}$$
- **Acceptable Bound ($P_{10} \le x < P_{25}$ or $P_{75} < x \le P_{90}$):** Score is scaled continuously between $70.0$ and $89.9$.
- **Outlier Penalty ($x < P_{10}$ or $x > P_{90}$):** Linear decay below $70.0$ proportional to distance over IQR:
  $$\text{Score} = \max\left(0.0, 70.0 - 70.0 \cdot \frac{\text{dist}}{\text{IQR}}\right)$$

#### Policy 2: Lower is Better (e.g., `head_knee_alignment`, `spine_lean`, `head_stability`)
- $x \le P_{25}$: Score $= 98.0$
- $P_{25} < x \le P_{50}$: Scaled from $98.0$ down to $90.0$
- $P_{50} < x \le P_{75}$: Scaled from $90.0$ down to $75.0$
- $P_{75} < x \le P_{90}$: Scaled from $75.0$ down to $60.0$
- $x > P_{90}$: Scaled below $60.0$ based on IQR distance

#### Policy 3: Higher is Better (e.g., `front_elbow_angle`, `follow_through_angle`, `follow_through_quality`)
- $x \ge P_{75}$: Score $= 98.0$
- $P_{50} \le x < P_{75}$: Scaled from $90.0$ up to $98.0$
- $P_{25} \le x < P_{50}$: Scaled from $75.0$ up to $90.0$
- $P_{10} \le x < P_{25}$: Scaled from $60.0$ up to $75.0$
- $x < P_{10}$: Scaled below $60.0$ based on IQR distance

### 17.2 Dynamic Occlusion Weight Renormalization
If a metric value is `None` (due to keypoint occlusion or low visibility), the scoring engine does NOT assign a zero. Instead, it dynamically redistributes that metric's weight across the remaining valid metrics in the category:
$$w_i' = \frac{w_i}{\sum_{j \in \text{valid}} w_j}$$
If an entire category has no valid metrics, the category is marked as excluded and overall category weights are renormalized across the remaining categories.

### 17.3 Fallback Heuristic Rule Engine
If reference calibration JSON files are missing or corrupted, the system activates `_evaluate_fallback_expert()` in `src/evaluation/scoring_engine.py`, which grades strokes against threshold bounds defined in `config.json`:
- `elbow_angle_threshold`: $120^\circ$
- `head_knee_distance_threshold`: $50\text{ px}$ (or normalized $< 0.3$)
- `foot_angle_lower_threshold`: $45^\circ$, `foot_angle_upper_threshold`: $135^\circ$
- `spine_lean_threshold`: $30^\circ$

---

# 18. Shot Classification / Evaluation

Shot classification is implemented in [`src/shot_analysis/shot_classifier.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/shot_analysis/shot_classifier.py).

### 18.1 Classification Methodology
AthleteRise employs an explainable, data-driven prototype classifier:
1. **Feature Extraction:** A 12-dimensional vector $\mathbf{x} \in \mathbb{R}^{12}$ is extracted from the contact window and temporal phases.
2. **Pooled Standard Deviation Normalization:** Each feature is scaled by pooled standard deviation $\hat{\sigma}_k$ across all reference samples:
   $$z_k = \frac{x_k - \mu_{c, k}}{\hat{\sigma}_k}$$
3. **Biomechanical Weighting:** Features with higher between-class variance are assigned higher weights:
   - Shoulder Rotation: $1.8$
   - Front Foot Direction: $1.8$
   - Wrist Horizontal Travel: $1.6$
   - Head Stability: $1.5$
   - Hip Rotation: $1.4$
   - Spine Lean: $1.2$
   - Head-Knee Alignment: $1.2$
   - Wrist Vertical Travel: $1.0$
   - Front Elbow Angle: $0.8$
   - Stride Length: $0.6$
   - Rotary Shoulder Diff: $0.2$
   - Follow-Through Angle: $0.1$
4. **Standardized Weighted Distance:**
   $$D(c) = \sqrt{\sum_{k=1}^{12} w_k \left( \frac{x_k - \mu_{c, k}}{\hat{\sigma}_k} \right)^2}$$
5. **Softmax Confidence Mapping:**
   $$P(c) = \frac{\exp(-D(c))}{\sum_{j} \exp(-D(j))}$$
6. **Confidence & Margin Gating:**
   Let $c_1$ be the top class candidate and $c_2$ the runner-up.
   $$\text{Predicted Shot} = \begin{cases} c_1 & \text{if } P(c_1) \ge 0.50 \text{ and } (P(c_1) - P(c_2)) \ge 0.12 \\ \text{UNKNOWN} & \text{otherwise} \end{cases}$$

If classified as `UNKNOWN`, the system does NOT score the stroke against any specific shot baseline. Instead, it displays the general biomechanical measurements without penalty.

---

# 19. Multi-Shot Analysis

The platform provides end-to-end multi-shot capability across Cover Drive, Straight Drive, and Pull Shot:

| Functional Stage | Cover Drive | Straight Drive | Pull Shot |
| :--- | :--- | :--- | :--- |
| **Primary Kinematic Focus** | Off-side flow, front arm reach | Down-the-pitch verticality | Cross-bat trunk uncoiling |
| **Phase Detection Cue** | Downswing front elbow inflection | Head-knee vertical alignment | Peak shoulder rotational velocity |
| **Reference Baseline** | `output/.../cover_drive_reference.json` | `output/.../straight_drive_reference.json`| `output/.../pull_shot_reference.json` |
| **Key Differentiating Feature**| $140^\circ-175^\circ$ foot, $144^\circ$ shoulder | $168^\circ$ foot, $0.11$ head-knee offset | $169^\circ$ shoulder rotation, compact stride |
| **Shot-Specific Output** | `cover_direction_alignment` | `straight_bat_presentation` | `rotary_trunk_uncoiling` |

---

# 20. Real-Time Analysis

The term "Real-Time" appears in the filename `cover_drive_analysis_realtime.py`. A technical audit reveals the following actual operational characteristics:

1. **Processing Architecture:** The pipeline executes a **two-pass offline batch processing loop** over cached video frames:
   - **Pass 1:** Frame extraction, CLAHE normalization, pose estimation, and visual bat/ball detection.
   - **Pass 2:** Handedness detection, sequence-level phase segmentation, feature extraction, shot classification, scoring, HUD video rendering, and JSON export.
2. **Latency:** Because phase segmentation and contact estimation rely on future temporal context (e.g., follow-through extension after impact), processing cannot occur strictly on an isolated single frame in real-time streaming without a rolling buffer latency of $\ge 20-30\text{ frames}$ ($\sim 700 - 1000\text{ ms}$).
3. **Execution Throughput:** On modern multi-core x86-64 CPUs with OpenCV multi-threading, Pass 1 processes at approximately $15 - 25\text{ FPS}$ on 720p video.

---

# 21. Reporting System

The reporting system comprises three modules:

1. **`MultiShotReportGenerator` ([`src/reporting/multi_shot_report.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/reporting/multi_shot_report.py)):**
   - Formats evaluation results into a human-readable text document (`output/evaluation_report.txt`).
   - Contains sections for Shot Classification Summary, Class Probabilities, Vision Tracking Diagnostics, Bat-Ball Contact Analysis, General Biomechanical Measurements, Shot-Specific Indicators, 5 Category Scores, and Coaching Recommendations.

2. **`ContactReporter` ([`src/reporting/contact_report.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/reporting/contact_report.py)):**
   - Generates dedicated diagnostics for contact status, evidence confidence scores, spatial proximity distances (pixels), and trajectory deflection angles.

3. **`ShotCalibrationReporter` / `CoverDriveCalibrationReporter` ([`src/report/calibration_reporter.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/report/calibration_reporter.py)):**
   - Generates multi-panel Matplotlib distribution figures (`{shot_type}_distribution.png`) across all three shot types:
     - `output/reference_calibration/cover_drive/cover_drive_distribution.png`
     - `output/reference_calibration/straight_drive/straight_drive_distribution.png`
     - `output/reference_calibration/pull_shot/pull_shot_distribution.png`
   - Each distribution figure includes 4 publication-quality panels:
     1. $P_{10}$ to $P_{90}$ reference range interval plot with empirical Mean (red dot) and Median (green square) across 10 biomechanical metrics.
     2. Front Elbow Angle comparison bar chart across all calibration videos vs. baseline mean and $P_{10}/P_{90}$ bounds.
     3. Metric correlation matrix heatmap (`coolwarm` colormap).
     4. Normalized biomechanical polar radar profile chart.
   - Generates standalone, styled HTML calibration reports (`{shot_type}_calibration_report.html`) for Cover Drive, Straight Drive, and Pull Shot containing metadata cards, outlier diagnostics, video tracking status, and interactive visualizations.

---

# 22. Frontend / Application

The frontend is an interactive multi-tab web application built using **Streamlit** in [`app.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/app.py).

### Tab 1: 🎥 Single Video Shot Analysis
- **Controls:** Shot Classification Mode dropdown (`Auto-Detect Shot`, `Cover Drive (Force Baseline)`, `Straight Drive (Force Baseline)`, `Pull Shot (Force Baseline)`); video uploader (`.mp4`, `.avi`); Run Batting Analysis button.
- **Video Display:** Side-by-side or stacked annotated video player with download button.
- **Metric Cards (Row 1):** Detected Shot, Overall Technique Score, Candidate & Margin.
- **Metric Cards (Row 2):** Bat–Ball Contact Status, Biomechanical Impact Frame, Visual Contact Frame.
- **Expanders:**
  - *Shot Classification Diagnostics:* Class probabilities, top candidate margin, LOO-CV validation metrics, and tracking quality metrics.
  - *Bat-Ball Contact Deep Dive:* Spatial proximity (px), temporal alignment, relative velocity, ball deflection, bat consistency.
- **Biomechanical Table:** All 10 metrics with user value, score, reference mean, and P10-P90 range.
- **Reference Calibration Baseline Expander:** Statistical summary table, empirical box plots, and the reference distribution dashboard suite for the identified shot.
- **Category Scores:** 5-category breakdown cards.
- **Coaching Insights:** Key strengths and targeted recommendations.

### Tab 2: 📐 Multi-Shot Reference Calibration
- **Controls:** Shot selector (Cover Drive, Straight Drive, Pull Shot); displays count and filenames of detected reference videos.
- **Run Calibration Button:** Executes batch calibration for the selected shot class and regenerates all baseline artifacts.
- **Output Table:** Displays updated 11-descriptor reference statistics (Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR, P10, P90).
- **Box Plot Suite:** Multi-metric overview grid and single-metric deep dive with raw data overlays.
- **Distribution Dashboard Suite:** Embeds the 4-panel `{shot_type}_distribution.png` dashboard with instant PNG and HTML report download buttons.
- **Diagnostic Matrix:** Displays cross-shot comparison code block.

### Tab 3: 🎯 Contact Validation & Annotation Mode
- **Human-in-the-Loop Review:** Lists existing ground-truth annotations from `data/contact_annotations/`.
- **Validation Status:** Displays detector validation state (`NOT YET VALIDATED` or empirical precision/recall/F1).
- **Candidate Frame Review:** Renders a 3-frame inspection strip (`Frame Before`, `Candidate Contact Frame`, `Frame After`).
- **Annotation Buttons:** `CONFIRM CONTACT`, `NO CONTACT`, and `UNCERTAIN` save labeled records to `data/contact_annotations/`.

---

# 23. Backend / Core Modules

| Module Path | Primary Responsibility | Input Data | Output Data |
| :--- | :--- | :--- | :--- |
| `src/preprocessing/video_preprocessing.py` | Video loading, resizing, CLAHE lightness normalization | Video file path (`.mp4`, `.avi`) | Frame generator `(frame_idx, frame)` |
| `src/pose_estimation/posture_tracker.py` | Keypoint tracking via MediaPipe with flow fallback | BGR video frames | 15-joint normalized coordinate dictionary |
| `src/biomechanics/metrics.py` | 10 core batting angles and handedness detection | Landmark sequences | Dict of instantaneous kinematic metrics |
| `src/biomechanics/shot_specific_metrics.py`| Computes specialized metrics per shot type | Contact metrics dict, shot type | Dict of shot-specific indicators |
| `src/shot_analysis/shot_features.py` | Extracts 12D feature vector and quality metadata | Processed frame metrics list | 12D numeric dict + quality reason dict |
| `src/shot_analysis/shot_classifier.py` | Prototype distance classification with LOO-CV | 12D feature vector | Predicted shot, margin, probabilities |
| `src/phase_detection/phase_factory.py` | Factory dispatching shot-specific phase detectors | Shot class string | `BasePhaseDetector` instance |
| `src/contact_detection/bat_ball_detector.py` | Segments bat and tracks ball trajectory | Video frames + wrist coordinates | Bounding boxes, segments, ball centers |
| `src/contact_detection/contact_estimator.py` | Multi-signal bat-ball contact decision engine | Pose impact frame, bat/ball tracks | Contact status, visual contact frame, signals |
| `src/calibration/base_calibration_engine.py` | Batch calibration & 14 statistical descriptors | List of reference video paths | Statistics database dictionary |
| `src/evaluation/base_scoring_engine.py` | Continuous percentile scoring & category weights | User metrics, reference baseline | Overall score, category scores, skill grade |
| `src/evaluation/feedback_engine.py` | Generates diagnostic coaching cues | Category and metric score dicts | Strengths and recommendations lists |
| `src/reporting/multi_shot_report.py` | Formats text evaluation reports | Evaluation results dictionary | Formatted string report |

---

# 24. Configuration

The configuration is maintained in [`config.json`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/config.json):

| Parameter | Current Value | Technical Meaning | Consuming Module |
| :--- | :---: | :--- | :--- |
| `elbow_angle_threshold` | `120` | Minimum lead elbow angle ($^\circ$) for expert fallback | `src/evaluation/scoring_engine.py` |
| `head_knee_distance_threshold` | `50` | Maximum pixel distance for head-knee alignment fallback | `src/evaluation/scoring_engine.py` |
| `foot_angle_lower_threshold` | `45` | Lower bound ($^\circ$) for front foot angle fallback | `src/evaluation/scoring_engine.py` |
| `foot_angle_upper_threshold` | `135` | Upper bound ($^\circ$) for front foot angle fallback | `src/evaluation/scoring_engine.py` |
| `spine_lean_threshold` | `30` | Maximum spine lean ($^\circ$) for balance fallback | `src/evaluation/scoring_engine.py` |
| `shots.cover_drive.reference_path` | `"data/reference_videos/cover_drive"` | Directory containing Cover Drive reference clips | `src/shot_analysis/shot_config.py` |
| `shots.cover_drive.reference_file` | `"output/.../cover_drive_reference.json"`| Calibrated Cover Drive baseline JSON destination | `src/shot_analysis/shot_config.py` |
| `shots.straight_drive.reference_path`| `"data/reference_videos/straight_drive"`| Directory containing Straight Drive reference clips | `src/shot_analysis/shot_config.py` |
| `shots.straight_drive.reference_file`| `"output/.../straight_drive_reference.json"`| Calibrated Straight Drive baseline JSON destination | `src/shot_analysis/shot_config.py` |
| `shots.pull_shot.reference_path` | `"data/reference_videos/pull_shot"` | Directory containing Pull Shot reference clips | `src/shot_analysis/shot_config.py` |
| `shots.pull_shot.reference_file` | `"output/.../pull_shot_reference.json"` | Calibrated Pull Shot baseline JSON destination | `src/shot_analysis/shot_config.py` |
| `classifier.confidence_threshold` | `0.50` | Minimum probability required to classify shot | `src/shot_analysis/shot_classifier.py` |
| `classifier.margin_threshold` | `0.12` | Minimum probability margin between Top 1 and Top 2 | `src/shot_analysis/shot_classifier.py` |
| `contact_detection.enabled` | `true` | Toggles bat-ball contact estimation engine | `cover_drive_analysis_realtime.py` |
| `contact_detection.unknown_threshold`| `0.50` | Confidence threshold below which contact is UNKNOWN | `src/contact_detection/contact_estimator.py` |
| `contact_detection.spatial_proximity_threshold`| `75.0` | Maximum pixel distance for bat-ball contact candidate | `src/contact_detection/contact_estimator.py` |
| `contact_detection.deflection_threshold_deg` | `25.0` | Minimum post-impact ball trajectory deflection ($^\circ$) | `src/contact_detection/contact_estimator.py` |

---

# 25. Scripts

The `scripts/` directory provides operational and diagnostic utilities:

- **`scripts/evaluate_dataset.py`:** Runs re-substitution evaluation across all 30 calibrated reference samples, outputting confusion matrices, precision, recall, and overall accuracy.
- **`scripts/evaluate_models.py`:** Benchmarks experimental machine-learning classifiers (Logistic Regression, SVM linear, SVM RBF, Random Forest) via Leave-One-Out Cross-Validation.
- **`scripts/compare_cover_straight.py`:** Compares feature vectors and distributions between Cover Drive and Straight Drive clips.
- **`scripts/test_weighted_classifier.py`:** Tests prototype classifier distance functions and softmax probability outputs across test vectors.
- **`scripts/inspect_stats.py`:** Reads calibrated reference JSONs and prints summary tables to console.
- **`scripts/inspect_per_video.py`:** Prints per-video extracted metrics for individual reference clips.
- **`scripts/check_aligned_tilt.py` & `check_shoulders.py`:** Diagnostic scripts verifying spine lean and shoulder rotation calculation parity.

---

# 26. Testing

The automated test suite is located in `tests/` and contains **35 verified unit tests**:

### 1. `tests/test_audit_architecture.py` (7 Tests)
- `test_continuous_scoring_policies`: Verifies mathematical curves for optimal, lower-is-better, and higher-is-better scoring policies.
- `test_dynamic_na_renormalization_within_category`: Verifies dynamic weight redistribution when a metric is `None`.
- `test_dynamic_na_renormalization_across_categories`: Verifies dynamic redistribution when an entire category is missing.
- `test_follow_through_angle_monotonicity`: Verifies that higher follow-through angles yield monotonically non-decreasing scores.
- `test_left_handed_stance_coordinate_inversion`: Verifies correct lead limb assignment for left-handed batters.
- `test_fallback_to_expert_rules_on_missing_reference`: Verifies graceful fallback to `config.json` rules when reference baseline is missing.
- `test_master_scoring_scale_alignment`: Verifies that overall scores map strictly to the defined project scoring grades.

### 2. `tests/test_audit_step19_pipeline.py` (11 Tests)
- Explicitly tests 11 operational scenarios:
  1. Known Cover Drive prototype $\rightarrow$ `cover_drive`
  2. Known Straight Drive prototype $\rightarrow$ `straight_drive`
  3. Known Pull Shot prototype $\rightarrow$ `pull_shot`
  4. Completely unrelated non-cricket features $\rightarrow$ `UNKNOWN`
  5. Insufficient pose tracking coverage $\rightarrow$ `UNKNOWN`
  6. Bat tracking unavailable but ball available $\rightarrow$ Contact `UNKNOWN`
  7. Ball tracking unavailable near contact $\rightarrow$ Contact `UNKNOWN`
  8. Bat and ball reliably meet $\rightarrow$ `CONTACT_DETECTED`
  9. Bat and ball reliably miss $\rightarrow$ `NO_CONTACT_DETECTED`
  10. Partial signals $\rightarrow$ Handled robustly
  11. Existing cover drive pipeline works without regression

### 3. `tests/test_multi_shot_analysis.py` (17 Tests)
- Tests multi-shot classification, empirical calibration output directories, zero cross-contamination between shot databases, bat/ball detector initialization, annotation saving and retrieval, and backward compatibility with `CoverDriveScoringEngine`.

*Execution command:*
```bash
python -m unittest discover tests
# Result: Ran 35 tests in 0.046s — OK
```

---

# 27. Evaluation

### 27.1 Shot Classification Evaluation

#### Re-substitution Evaluation (30 Reference Samples)
Evaluated via `scripts/evaluate_dataset.py` across all 30 calibrated reference videos:
- **Cover Drive ($N=10$):** $10/10$ Correct ($100.0\%$)
- **Straight Drive ($N=10$):** $10/10$ Correct ($100.0\%$)
- **Pull Shot ($N=10$):** $10/10$ Correct ($100.0\%$)
- **Overall Re-substitution Accuracy:** $\mathbf{100.0\%}$ ($30/30$)
- **Per-Class F1-Scores:** Cover Drive $1.000$, Straight Drive $1.000$, Pull Shot $1.000$.

#### Leave-One-Out Cross-Validation (LOO-CV)
Evaluated via `ShotClassifier().evaluate_loo_cross_validation()`:
- **Total Samples:** $30$ ($10$ clips per class)
- **LOO-CV Accuracy:** $\mathbf{50.0\%}$ ($15/30$)
- **Macro F1-Score:** $\mathbf{49.9\%}$
- **Per-Class Metrics:**
  - Cover Drive: Precision $54.5\%$, Recall $60.0\%$, F1-Score $57.1\%$
  - Straight Drive: Precision $40.0\%$, Recall $40.0\%$, F1-Score $40.0\%$
  - Pull Shot: Precision $55.6\%$, Recall $50.0\%$, F1-Score $52.6\%$
- **Technical Context:** Because $N=10$ per class is a small sample size, removing one sample noticeably shifts the prototype mean and variance of that class. This empirical result is explicitly displayed in the Streamlit UI with the notice: *"Results are preliminary due to the small reference dataset (N=10 per shot class)."*

### 27.2 Bat–Ball Contact Detection Validation
- **Validation Status:** Evaluated via `ContactAnnotationManager().evaluate_detector_against_annotations()`.
- **Current Status:** `NOT YET VALIDATED`.
- **Reason:** The ground-truth annotation repository (`data/contact_annotations/`) currently contains human-in-the-loop candidate records but lacks a standardized large-scale benchmark dataset ($N > 100$). The system accurately reports this status in the UI rather than claiming unvalidated accuracy.

---

# 28. Results

| Evaluation Area | Measured Value | Validation Status |
| :--- | :--- | :--- |
| **Architectural Unit Tests** | $35 / 35$ Passed ($100\%$) | ✅ Formally Validated |
| **Shot Classifier Re-substitution**| $30 / 30$ ($100.0\%$) | ✅ Formally Validated on Reference Set |
| **Shot Classifier LOO-CV** | $50.0\%$ Accuracy, $49.9\%$ Macro F1 | ✅ Formally Validated (Small $N=30$) |
| **Pose Estimation Tracking** | $100\%$ pass on reference set | ✅ Formally Validated on Reference Set |
| **Bat–Ball Contact Precision/Recall**| Not currently validated | ⚠️ Not currently validated (Dataset pending) |
| **Large-Scale External Cohort** | Not currently validated | ⚠️ Requires field trials with match footage |

---

# 29. Performance

- **Test Suite Execution:** 35 unit tests execute in **$0.046\text{ seconds}$**.
- **Batch Reference Calibration:** Calibrating all 30 reference videos across 3 shot classes takes approximately **$45 - 60\text{ seconds}$** total on an Intel Core i7 / AMD Ryzen multi-core workstation.
- **Single Video Processing:** A standard 5-second 30 FPS video ($150\text{ frames}$) is fully processed, scored, and rendered to annotated MP4 in approximately **$6 - 10\text{ seconds}$** ($\sim 15-25\text{ FPS}$).
- **Hardware Benchmarks:** *No validated multi-hardware benchmark suite is currently available.*

---

# 30. Error Handling

AthleteRise implements defensive error handling across all execution paths:

1. **Missing MediaPipe Fallback:** If `mediapipe.python.solutions.pose` cannot be imported, the system attempts `mediapipe.tasks` PoseLandmarker, then falls back to optical flow propagation and silhouette bounding box estimates.
2. **Missing Reference Baseline Handling:** If reference JSON baselines are absent, `MultiShotScoringEngine` catches `FileNotFoundError` and activates `_evaluate_fallback_expert()`, grading the stroke via `config.json` threshold heuristics without crashing.
3. **Occluded Keypoint Handling:** If landmarks fail confidence thresholds ($< 0.35 - 0.50$), metric calculations return `None`. The scoring engine triggers dynamic N/A renormalization, redistributing weight rather than penalizing with zero.
4. **Unknown Shot Handling:** If classifier confidence is $< 0.50$ or margin is $< 0.12$, the stroke is marked as `UNKNOWN`. Shot-specific scoring is withheld, and raw biomechanical measurements are displayed with explanatory captions.
5. **Video File Corruption:** `VideoPreprocessor` verifies `cap.isOpened()` and throws a descriptive `ValueError` if the file cannot be decoded.

---

# 31. Limitations

To maintain scientific integrity, the following limitations of the current codebase must be recognized:

1. **Monocular 2D Depth Ambiguity:** A single 2D camera cannot measure true 3D spatial joint coordinates. Out-of-plane movements (such as bat swing depth along the Z-axis) are projected onto the 2D image plane.
2. **Small Reference Dataset ($N=10$ per class):** While $N=10$ establishes a functional baseline for demonstration, it does not represent the full statistical diversity of international cricket technique.
3. **Ball Tracking Under Severe Motion Blur:** High-speed deliveries ($> 120\text{ km/h}$) captured at standard consumer frame rates ($30\text{ FPS}$) exhibit significant ball motion blur, resulting in intermittent ball detection near contact.
4. **Self-Occlusion:** In side-on recordings, the batter's lead arm and torso frequently occlude the rear elbow and hip during follow-through.
5. **Fixed Batting Stroke Scope:** The active codebase supports Cover Drive, Straight Drive, and Pull Shot. Horizontal sweeps, cuts, and bowling actions are not supported.

---

# 32. Installation

### 32.1 Prerequisites
- Operating System: Windows 10/11, macOS (x86_64 or Apple Silicon), or Linux (Ubuntu 20.04+).
- Python: **Version 3.10.x** (Strictly recommended due to compiled C++ protobuf bindings in MediaPipe).

### 32.2 Virtual Environment Setup

```bash
# Clone or navigate to the repository
cd AI-Powered-Cricket-Analytics-main

# Create virtual environment
python -m venv cricket_analytics

# Activate environment
# On Windows PowerShell:
.\cricket_analytics\Scripts\Activate.ps1
# On Linux / macOS:
source cricket_analytics/bin/activate
```

### 32.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 33. Configuration

Users can customize analysis parameters in `config.json`:
- **Change classifier thresholds:** Adjust `classifier.confidence_threshold` (default `0.50`) and `classifier.margin_threshold` (default `0.12`).
- **Modify contact thresholds:** Adjust `contact_detection.spatial_proximity_threshold` (default `75.0` pixels) and `contact_detection.deflection_threshold_deg` (default `25.0` degrees).
- **Update expert fallback limits:** Adjust `elbow_angle_threshold`, `head_knee_distance_threshold`, and `spine_lean_threshold`.

---

# 34. Running the Project

### 34.1 Launch Streamlit Web Application (Recommended)
```bash
streamlit run app.py
```
*Access via browser at `http://localhost:8501`.*

### 34.2 Run Standalone CLI Video Analysis
```bash
# Analyze a video with automatic shot detection
python cover_drive_analysis_realtime.py --video data/reference_videos/cover_drive/cover_0001.avi

# Override shot baseline to Straight Drive
python cover_drive_analysis_realtime.py --video data/reference_videos/straight_drive/straight_0001.avi --shot straight_drive
```

### 34.3 Run Reference Dataset Recalibration
```bash
python calibrate_references.py
```

### 34.4 Run Automated Unit Test Suite
```bash
python -m unittest discover tests
```

### 34.5 Run Empirical Dataset Evaluation
```bash
python scripts/evaluate_dataset.py
```

---

# 35. Complete Example Workflow

1. **Step 1 — Video Ingestion:** Athlete records a batting shot on a mobile phone (side-on angle, 720p or 1080p, 30-60 FPS) and uploads it via Tab 1 in `app.py`.
2. **Step 2 — Preprocessing:** `VideoPreprocessor` applies CLAHE lightness equalization in the LAB color space to normalize pitch lighting.
3. **Step 3 — Skeletal Tracking:** `VisionPostureTracker` tracks 15 anatomical keypoints across all frames.
4. **Step 4 — Stance & Handedness:** `detect_handedness` identifies whether the batter is right- or left-handed.
5. **Step 5 — Feature Extraction & Shot Classification:** `ShotFeatureExtractor` extracts the 12D feature vector; `ShotClassifier` determines the shot class (e.g., Cover Drive at $98.8\%$ confidence).
6. **Step 6 — Phase Detection:** `PhaseDetectorFactory` creates `CoverDrivePhaseDetector`, locating impact frame $t_p$ and segmenting Stance, Downswing, Contact Window, and Follow-Through.
7. **Step 7 — Bat–Ball Contact Detection:** `ContactEstimator` correlates bat segment tracking with ball deflection, estimating visual contact at frame $t_c$.
8. **Step 8 — Scoring:** `MultiShotScoringEngine` selects `cover_drive_reference.json`, compares user medians against $P_{10} - P_{90}$ curves, applies dynamic N/A renormalization, and calculates category scores.
9. **Step 9 — Output Delivery:** The UI displays the annotated MP4 video, metric evaluation tables, category score cards, and explainable coaching recommendations.

---

# 36. Current Implementation Status

| Feature / Component | Status | Codebase Evidence |
| :--- | :---: | :--- |
| **Monocular Video Preprocessing** | ✅ Implemented | [`src/preprocessing/video_preprocessing.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/preprocessing/video_preprocessing.py) |
| **MediaPipe 15-Keypoint Tracking**| ✅ Implemented | [`src/pose_estimation/posture_tracker.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/pose_estimation/posture_tracker.py) |
| **Automated Handedness Detection**| ✅ Implemented | [`src/biomechanics/metrics.py:L20-L52`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/biomechanics/metrics.py#L20-L52) |
| **10 Core Biomechanical Metrics** | ✅ Implemented | [`src/biomechanics/metrics.py:L96-L220`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/biomechanics/metrics.py#L96-L220) |
| **Shot-Specific Biomechanics** | ✅ Implemented | [`src/biomechanics/shot_specific_metrics.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/biomechanics/shot_specific_metrics.py) |
| **12D Kinematic Feature Extraction**| ✅ Implemented | [`src/shot_analysis/shot_features.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/shot_analysis/shot_features.py) |
| **Prototype Shot Classifier** | ✅ Implemented | [`src/shot_analysis/shot_classifier.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/shot_analysis/shot_classifier.py) |
| **Multi-Shot Phase Segmentation** | ✅ Implemented | [`src/phase_detection/phase_factory.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/phase_detection/phase_factory.py) |
| **Bat & Ball Tracking** | ✅ Implemented | [`src/contact_detection/bat_ball_detector.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/contact_detection/bat_ball_detector.py) |
| **Bat–Ball Contact Estimator** | ✅ Implemented | [`src/contact_detection/contact_estimator.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/contact_detection/contact_estimator.py) |
| **Empirical Calibration ($N=30$)** | ✅ Implemented | [`src/calibration/base_calibration_engine.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/calibration/base_calibration_engine.py) |
| **Continuous Percentile Scoring** | ✅ Implemented | [`src/evaluation/base_scoring_engine.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/evaluation/base_scoring_engine.py) |
| **Dynamic Occlusion Renormalization**| ✅ Implemented | [`src/evaluation/scoring_engine.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/evaluation/scoring_engine.py) |
| **Fallback Heuristic Rule Engine** | ✅ Implemented | [`src/evaluation/scoring_engine.py:L392-L462`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/evaluation/scoring_engine.py#L392-L462) |
| **Streamlit 3-Tab Web GUI** | ✅ Implemented | [`app.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/app.py) |
| **HUD Video Annotation Renderer** | ✅ Implemented | [`cover_drive_analysis_realtime.py:L445-L596`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/cover_drive_analysis_realtime.py#L445-L596) |
| **Structured Multi-Shot Reporting**| ✅ Implemented | [`src/reporting/multi_shot_report.py`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/src/reporting/multi_shot_report.py) |
| **35-Case Automated Unit Test Suite**| ✅ Implemented | [`tests/`](file:///c:/Users/Shidd/.gemini/antigravity/playground/AI-Powered-Cricket-Analytics-main/tests/) |
| **Contact Ground Truth Annotator**| 🟡 Partial | Tab 3 functional; large benchmark pending |
| **Generative LLM Coaching Cues** | 🟡 Partial | Template cues implemented; generative commentary planned |
| **Fast Bowling Delivery Phasing** | 🔴 Planned | Not implemented (Batting only) |
| **Deep Spatio-Temporal GCN (CTR-GCN)**| 🔴 Planned | Not implemented |
| **FAISS High-Dimensional Vector Index**| 🔴 Planned | Not implemented |
| **RTMPose-M Keypoint Model** | 🔴 Planned | Not implemented |

---

# 37. Research / Technical Contribution

AthleteRise makes the following technical contributions to sports computer vision:
1. **Scale-Invariant Kinematic Formulation:** Formulates batting metrics using joint angle trigonometry, torso-length normalized horizontal offsets, and leg-length normalized stride ratios, eliminating camera distance sensitivity.
2. **Empirical Distribution Baseline over Single-Target Mimicry:** Replaces rigid single-expert coordinate matching with continuous empirical percentile bands ($P_{10} - P_{90}$), accommodating anthropometric variance across athletes.
3. **Decoupled Visual Contact vs Biomechanical Impact:** Explicitly separates physical bat-ball contact from kinematic body inflection, eliminating ambiguity in contact detection.
4. **Dynamic Occlusion Weight Renormalization:** Introduces an occlusion-aware weighting algorithm that dynamically recalculates category scores when limbs are obscured.
5. **Explainable Centroid Prototype Classification:** Provides fully transparent shot classification with explicit feature contributions, class probabilities, and margin gating rather than uninterpretable black-box deep embeddings.

---

# 38. Future Work

1. **Expansion to Comprehensive Stroke Library:** Addition of Cut, Hook, Sweep, On-Drive, and Back-Foot Defense reference baselines.
2. **Fast Bowling Biomechanics Module:** Segmentation of the bowling action into Run-up, Bound, Back-Foot Impact, Front-Foot Landing, Ball Release, and Follow-Through, tracking lumbar hyperextension and knee bracing angle.
3. **High-Speed Camera Integration:** Support for 120-240 FPS high-frame-rate inputs to eliminate ball motion blur.
4. **Graph Convolutional Network Classification:** Training ST-GCN or CTR-GCN architectures on joint coordinate sequences for automated action segmentation.
5. **Mobile On-Device Deployment:** Quantization and deployment to iOS (CoreML) and Android (ONNX Runtime / TFLite) for instant on-pitch processing.

---

# 39. Reproducibility

To independently reproduce the system and verify all results:
1. Clone the repository and install requirements on Python 3.10.
2. Run `python -m unittest discover tests` — verify that all **35 tests pass**.
3. Run `python calibrate_references.py` — verify that all 3 reference JSON baselines are generated in `output/reference_calibration/` and match the values in Section 16.3.
4. Run `python scripts/evaluate_dataset.py` — verify $100\%$ re-substitution accuracy across the 30 reference samples.
5. Launch `streamlit run app.py` — upload any reference video from `data/reference_videos/` and verify that the system correctly classifies the shot, generates annotated video, and produces scoring cards.

---

# 40. Final Technical Summary

**AthleteRise (AI-Powered Cricket Analytics)** represents an end-to-end computer-vision system for markerless cricket batting kinematics. Operating on monocular 2D video, it combines lighting-invariant preprocessing (CLAHE LAB), robust skeletal tracking (MediaPipe Pose with optical-flow fallback), explainable multi-shot prototype classification, modular bat-ball contact estimation, and empirical reference calibration ($N=30$). By replacing subjective coaching impressions with scale-invariant telemetry, continuous percentile evaluation, and dynamic occlusion resilience, AthleteRise delivers accessible, objective sports biomechanics for cricketers and coaches worldwide.

---

# 41. Appendix

### A. Core Mathematical Formulas

$$\text{Interior Joint Angle: } \theta = |\operatorname{atan2}(c_y - b_y, c_x - b_x) - \operatorname{atan2}(a_y - b_y, a_x - b_x)| \times \frac{180^\circ}{\pi}$$

$$\text{Torso Normalized Head-Knee Offset: } \text{HK} = \frac{|x_{\text{nose}} - x_{\text{knee}}|}{\|\mathbf{p}_{\text{mid\_shoulder}} - \mathbf{p}_{\text{mid\_hip}}\|_2 + \epsilon}$$

$$\text{Normalized Stride Ratio: } \text{SR} = \frac{\|\mathbf{p}_{\text{front\_ankle}} - \mathbf{p}_{\text{back\_ankle}}\|_2}{\|\mathbf{p}_{\text{front\_hip}} - \mathbf{p}_{\text{front\_ankle}}\|_2 + \epsilon}$$

$$\text{Continuous Score (IQR Bound): } S(x) = \max\left(0.0, 70.0 - 70.0 \cdot \frac{|x - P_{\text{bound}}|}{\text{IQR}}\right)$$

$$\text{Dynamic Category Score: } C_k = \frac{\sum_{i \in \text{valid}} w_i \cdot S_i}{\sum_{i \in \text{valid}} w_i}$$

$$\text{Overall Technique Score: } \text{Score}_{\text{overall}} = \frac{\sum_{k \in \text{valid}} W_k \cdot C_k}{\sum_{k \in \text{valid}} W_k}$$

### B. Project Scoring Scale & Skill Tiers
- **$90.0 - 100.0$:** Advanced / Elite Execution
- **$75.0 - 89.9$:** Above Average / Proficient Execution
- **$60.0 - 74.9$:** Intermediate Execution
- **$45.0 - 59.9$:** Developing Technique
- **$< 45.0$:** Beginner / Foundational Refinement

### C. Hierarchical Weighting Schema
- **Swing Control ($25\%$):** Front Elbow Angle ($50\%$), Shoulder Rotation ($25\%$), Hip Rotation ($25\%$)
- **Head Position ($20\%$):** Head-Knee Alignment ($60\%$), Head Stability ($40\%$)
- **Footwork ($20\%$):** Front Foot Direction ($60\%$), Stride Length ($40\%$)
- **Balance ($20\%$):** Spine Lean Tilt ($60\%$), Head Stability ($40\%$)
- **Follow-Through ($15\%$):** Follow-Through Angle ($50\%$), Follow-Through Quality ($50\%$)

### D. Key Verification Commands
```bash
# Run complete test suite (35 unit tests)
python -m unittest discover tests

# Launch Streamlit web dashboard
streamlit run app.py

# Re-run empirical calibration
python calibrate_references.py

# Run standalone video analysis
python cover_drive_analysis_realtime.py --video data/reference_videos/cover_drive/cover_0001.avi
```
