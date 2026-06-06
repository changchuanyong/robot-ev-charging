# Project Structure

```text
robot-ev-charging/
├─ config/                 Runtime, training, camera, and model-point configs
├─ docs/                   Paper/report material and generated publication figures
│  └─ chapter3/
├─ live/
│  ├─ cpp/                 Kinect live capture source
│  └─ python/              Online vision pipeline
├─ train/
│  ├─ cpp/                 Training image acquisition source
│  └─ python/              Dataset conversion, checks, and YOLO training
├─ yolo_port/              YOLO dataset, tracked images and labels
├─ dataset/live/           Runtime frame exchange and debug outputs, ignored by Git
├─ artifacts/              Local audit images and temporary generated files, ignored by Git
└─ runs/                   YOLO training outputs, ignored by Git
```

## Rules

- Keep source code in `train/`, `live/`, or `docs/` scripts.
- Keep reusable configuration in `config/`.
- Keep YOLO training data in `yolo_port/images` and `yolo_port/labels`.
- Name YOLO samples by split: `train_port_0001.*`, `train_port_0002.*`, `val_port_0001.*`.
- Keep runtime outputs under `dataset/live/`; do not commit them.
- Keep temporary visual audits under `artifacts/`; do not commit them.
- Keep large model weights and training runs out of Git.
