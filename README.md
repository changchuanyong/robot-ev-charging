# robot-ev-charging

Electric vehicle charging-port vision pipeline. The project covers dataset preparation, YOLO-based port detection, ROI post-processing, pin-hole center extraction, and PnP pose estimation for robotic EV charging experiments.

## Visual Overview

### YOLO ROI Detection

![YOLO ROI detection](docs/assets/readme/roi_detection.jpg)

The live RGB frame is processed by YOLO to locate the charging port. The green box is the detector output and the yellow box is the padded ROI used by the downstream pipeline.

### Edge Extraction

![Edge overlay](docs/assets/readme/edge_overlay.jpg)

The ROI is enhanced and converted into an edge map. The current edge stage uses adaptive thresholds and a structure mask to keep the port area while suppressing unrelated outer edges.

### Candidate Pin Centers

![Candidate contours](docs/assets/readme/candidate_contours.jpg)

Contours are filtered by geometry, duplicate distance, and the standard charging-port layout prior. The final candidates are fitted to center points for PnP.

### PnP Pose Result

![PnP pose result](docs/assets/readme/pose_result.jpg)

The ROI points are mapped back to the original 1920x1080 image coordinates before solving PnP with the configured RGB camera intrinsics.

## Project Structure

```text
robot-ev-charging/
├─ config/
│  ├─ camera_intrinsics.json
│  ├─ charging_port_model.csv
│  ├─ live_pipeline_config.json
│  └─ train_config.json
├─ train/
│  ├─ python/
│  │  ├─ convert_labelme_to_yolo_detect.py
│  │  ├─ check_yolo_dataset.py
│  │  └─ train_port.py
│  └─ cpp/
├─ live/
│  ├─ cpp/
│  │  ├─ kinect_live_capture_atomic.cpp
│  │  └─ read_kinect_intrinsics.cpp
│  └─ python/
│     ├─ launch_pipeline.py
│     ├─ roi.py
│     ├─ enhance.py
│     ├─ Canny.py
│     ├─ latest_candidate_contours.py
│     ├─ candidate_types.py
│     ├─ layout_prior.py
│     ├─ center_fit.py
│     ├─ pnp2.py
│     ├─ pipeline_quality.py
│     └─ post_vis_watch.py
├─ yolo_port/
│  ├─ images/train, images/val
│  ├─ labels/train, labels/val
│  └─ dataset.yaml
├─ dataset/live/        # runtime inputs/outputs, ignored by Git
├─ docs/
└─ runs/                # YOLO training output, ignored by Git
```

## Environment

Recommended environment: Windows, Python 3.9+, Kinect SDK v2 if live capture is needed.

```powershell
python -m venv .yolo_env
.yolo_env\Scripts\activate
pip install -r requirements.txt
```

Main Python dependencies:

- `ultralytics`
- `torch`
- `opencv-python`
- `numpy`

## Dataset And Training

Labelme images and annotations live under:

```text
yolo_port/images/train
yolo_port/images/val
```

Convert Labelme annotations to YOLO labels:

```powershell
.yolo_env\Scripts\python.exe train\python\convert_labelme_to_yolo_detect.py
```

Check the YOLO dataset:

```powershell
.yolo_env\Scripts\python.exe train\python\check_yolo_dataset.py
```

Train the detector:

```powershell
.yolo_env\Scripts\python.exe train\python\train_port.py
```

Default trained weight path:

```text
runs/detect_retrain/weights/best.pt
```

## Live Pipeline

Start the full live pipeline:

```powershell
.yolo_env\Scripts\python.exe live\python\launch_pipeline.py
```

Pipeline configuration:

```text
config/live_pipeline_config.json
```

Main stages:

1. `roi.py`
   Detects the charging port and crops a padded ROI.
2. `enhance.py`
   Applies adaptive gamma, illumination normalization, controlled CLAHE, denoising, and limited sharpening.
3. `Canny.py`
   Generates the edge image. `VISION_EDGE_METHOD` can be set to `canny`, `morph`, or `hybrid`.
4. `latest_candidate_contours.py`
   Filters contours, applies the layout prior, fits centers, and exports center data.
5. `pnp2.py`
   Maps ROI points back to the original image and solves PnP.
6. `pipeline_quality.py`
   Writes a quality report for ROI, edges, candidates, and PnP.

## Runtime Outputs

Runtime files are written to `dataset/live/` and are ignored by Git:

```text
latest.jpg                       # captured RGB frame
latest_vis.jpg                   # YOLO/ROI visualization
latest_roi.jpg                   # cropped ROI
latest_roi_meta.json             # ROI metadata and original-image offset
latest_roi_enhanced.jpg          # enhanced ROI
latest_roi_enhance_metrics.json  # enhancement statistics
latest_edges.jpg                 # edge image
latest_edges_vis.jpg             # edge overlay
latest_candidate_contours.jpg    # contour/layout visualization
latest_fitted_centers.jpg        # fitted center visualization
latest_fitted_centers.json       # fitted center data
latest_pose.json                 # PnP pose result
latest_pose_vis.jpg              # PnP visualization
latest_pipeline_quality.json     # pipeline quality summary
```

## Quality Report

Run the quality checker after the pipeline has produced outputs:

```powershell
.yolo_env\Scripts\python.exe live\python\pipeline_quality.py
```

It reports:

- ROI detection score and crop size
- edge density
- candidate count, layout distance, and fit shift
- PnP reprojection error and camera-intrinsics status

Example current result:

```text
overall      : warn
roi          : ok
edges        : ok
candidates   : ok count=9
pose         : warn reproj≈10.8px
```

## Camera Intrinsics

`config/camera_intrinsics.json` currently contains a temporary Kinect v2 RGB 1920x1080 approximation:

```text
fx = 1081.37
fy = 1081.37
cx = 959.5
cy = 539.5
```

This is still marked as approximate. For accurate pose estimation, replace it with per-device RGB camera calibration.

Important detail: PnP uses original-image coordinates. ROI center points are mapped back to the 1920x1080 frame before calling PnP, so `camera_intrinsics.json` must describe the original RGB image, not the cropped ROI.

## Kinect Intrinsics Tool

`live/cpp/read_kinect_intrinsics.cpp` can read Kinect v2 depth intrinsics through Microsoft Kinect SDK v2:

```powershell
live\cpp\read_kinect_intrinsics.exe
```

Depth intrinsics are not RGB intrinsics. The current PnP pipeline uses RGB image points, so depth intrinsics should not be copied into `camera_intrinsics.json`.

## Useful Commands

Run candidate extraction and PnP once without blocking OpenCV windows:

```powershell
$env:VISION_PIPELINE_MODE='1'
.yolo_env\Scripts\python.exe live\python\latest_candidate_contours.py
.yolo_env\Scripts\python.exe live\python\pnp2.py
```

Keep debug windows open manually:

```powershell
$env:VISION_WAIT_FOR_KEY='1'
.yolo_env\Scripts\python.exe live\python\pnp2.py
```

Syntax check:

```powershell
.yolo_env\Scripts\python.exe -m py_compile live\python\latest_candidate_contours.py live\python\layout_prior.py live\python\center_fit.py live\python\pnp2.py
```

## Known Limitations

- RGB camera intrinsics still need real per-device calibration.
- The current validation is mainly based on existing live samples; new lighting and viewpoints should be tested.
- Kinect v2 depth data can help as a distance sanity check, but it is not used as the main pin-center detection signal.
- Some legacy source comments may still contain encoding artifacts and can be cleaned later.

## Git Ignore Policy

The following are not committed:

- `dataset/live/`
- `artifacts/`
- `runs/`
- `*.pt`
- Python cache and YOLO cache
- C/C++ build outputs such as `*.exe`, `*.obj`, `*.pdb`, `*.ilk`
