"""python -m vision.run --config config/realsense_config.json [--image image.jpg --camera-config camera.json]."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid

from .contracts import FramePacket, atomic_json, empty_observation
from .pipeline import PosePipeline, ROOT


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/realsense_config.json")
    parser.add_argument("--image", type=Path, help="Offline single frame; never accepted as live robot input")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=0, help="0 means continuous camera processing")
    parser.add_argument("--export-intrinsics", type=Path, help="Save actual color-stream intrinsics on camera startup")
    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    output = args.output or ROOT / config["output"]
    target = str(config.get("target_id", ""))
    atomic_json(output, empty_observation(None, target, "vision_starting"))
    try:
        pipeline = PosePipeline(config)
        config = pipeline.config
        if args.image:
            if not args.camera_config:
                raise ValueError("--image requires the image's --camera-config")
            import cv2
            image = cv2.imread(str(args.image))
            if image is None:
                raise ValueError("offline_image_unreadable")
            camera = json.loads(args.camera_config.read_text(encoding="utf-8-sig"))
            now = time.monotonic()
            packet = FramePacket(image, "offline:" + uuid.uuid4().hex, now, now, "offline", False, camera)
            result = pipeline.process(packet)
            atomic_json(output, result)
            print(json.dumps({"valid": result["valid"], "source": "offline", "reason": result["reason"],
                              "output": str(output)}, ensure_ascii=False))
            return 0 if result["valid"] else 2
        from .realsense import RealSenseCamera
        with RealSenseCamera(config) as camera:
            if args.export_intrinsics:
                atomic_json(args.export_intrinsics, camera.camera)
            count = 0
            while args.frames == 0 or count < args.frames:
                packet = camera.read()
                result = pipeline.process(packet)
                atomic_json(output, result)
                print(json.dumps({"frame_id": result["frame_id"], "valid": result["valid"],
                                  "reason": result["reason"]}, ensure_ascii=False), flush=True)
                count += 1
        atomic_json(output, empty_observation(None, target, "camera_stopped"))
        return 0
    except KeyboardInterrupt:
        atomic_json(output, empty_observation(None, target, "camera_stopped_by_user"))
        return 130
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        atomic_json(output, empty_observation(None, target, reason))
        print(reason)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
