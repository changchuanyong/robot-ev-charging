"""Collect original D415/D435 color images with per-session provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .contracts import intrinsics_signature, validate_camera
from .rgb_controls import RGBControls, validate_rgb_options
from .win_privilege import ensure_camera_privilege


ROOT = Path(__file__).resolve().parents[1]
WINDOW = "RealSense RGB data collection"


def list_devices(rs=None) -> list[dict]:
    if rs is None:
        import pyrealsense2 as rs
        ensure_camera_privilege(sdk=rs)
    context = rs.context()
    return [{"name": device.get_info(rs.camera_info.name),
             "serial": device.get_info(rs.camera_info.serial_number)}
            for device in context.query_devices()]


def validate_config(config: dict) -> None:
    if "rgb_options" in config:
        validate_rgb_options(config["rgb_options"])
    if config.get("device_model") not in {"D415", "D435"}:
        raise ValueError("数据采集配置的 device_model 必须明确为 D415 或 D435")
    if not isinstance(config.get("serial"), str) or not config["serial"].strip():
        raise ValueError("请用 --serial 指定完整序列号；--list-devices 可查看设备")
    for name in ("width", "height", "fps", "timeout_ms", "warmup_frames"):
        value = config.get(name)
        minimum = 0 if name == "warmup_frames" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"无效采集参数: {name}")


class CaptureColorCamera:
    """Independent data acquisition; never publishes a robot pose."""
    def __init__(self, config: dict):
        validate_config(config)
        self.config = dict(config)
        self.pipeline = None
        self.rgb_controls = None

    def _start_rgb_controls(self, device, rs):
        if "rgb_options" in self.config:
            self.rgb_controls = RGBControls(device, rs)
            self.rgb_controls.apply(self.config["rgb_options"])

    def __enter__(self):
        import pyrealsense2 as rs
        ensure_camera_privilege(sdk=rs)
        serial = self.config["serial"]
        matches = [device for device in list_devices(rs) if device["serial"] == serial]
        if len(matches) != 1:
            raise ValueError("未找到指定序列号的相机，请检查 USB 连接和 --list-devices")
        model = self.config["device_model"]
        if matches[0]["name"].split()[-1] != model:
            raise ValueError(f"该序列号不是指定的 {model}，数据采集入口拒绝启动")
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, self.config["width"], self.config["height"],
                             rs.format.bgr8, self.config["fps"])
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(config)
            device = profile.get_device()
            actual_serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            if actual_serial != serial or name.split()[-1] != model:
                raise ValueError("启动后的相机身份与指定型号/序列号不一致")
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color.get_intrinsics()
            if (intrinsics.width, intrinsics.height, color.fps(), color.format()) != (
                    self.config["width"], self.config["height"], self.config["fps"], rs.format.bgr8):
                raise ValueError("实际彩色流配置与请求的分辨率、帧率或格式不一致")
            self.camera = {
                "stream": "color", "serial": actual_serial, "device_name": name, "device_model": model,
                "image_width": intrinsics.width, "image_height": intrinsics.height,
                "fps": color.fps(), "pixel_format": "bgr8",
                "camera_matrix": [[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]],
                "dist_coeffs": list(intrinsics.coeffs), "distortion_model": intrinsics.model.name,
                "source": "device_color_stream", "validated": False, "version": None,
                "is_placeholder": False, "is_approximate": False,
            }
            validate_camera(self.camera, (intrinsics.height, intrinsics.width, 3))
            self.camera["intrinsics_sha256"] = intrinsics_signature(self.camera)
            self._start_rgb_controls(device, rs)
            for _ in range(self.config["warmup_frames"]):
                self.read()
        except BaseException:
            self.close()
            raise
        return self

    def read(self) -> tuple[np.ndarray, dict]:
        if self.pipeline is None:
            raise RuntimeError("采集相机尚未启动")
        frames = self.pipeline.wait_for_frames(self.config["timeout_ms"])
        received_at = datetime.now(timezone.utc).isoformat()
        color = frames.get_color_frame()
        if not color:
            raise ValueError("采集相机没有返回彩色帧")
        image = np.asanyarray(color.get_data()).copy()
        metadata = {"frame_number": color.get_frame_number(),
                    "device_timestamp_ms": color.get_timestamp(),
                    "timestamp_domain": color.get_frame_timestamp_domain().name,
                    "received_at_utc": received_at}
        if self.rgb_controls is not None:
            metadata["rgb_settings"] = self.rgb_controls.frame_settings(color)
        return image, metadata

    def close(self):
        pipeline, self.pipeline = self.pipeline, None
        try:
            if self.rgb_controls is not None:
                self.rgb_controls.restore()
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass

    def __exit__(self, *_):
        self.close()


# Keep the previous import working; model identity still comes from explicit config.
D415ColorCamera = CaptureColorCamera


def register_saved_rgb(session_path: Path, saved: dict, library_root: Path | None) -> bool:
    """Register committed raw RGB; library errors never roll back a capture."""
    if library_root is None:
        return True
    try:
        from .training_library import register_image

        session_path = Path(session_path).resolve()
        manifest_path = session_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_path = (session_path / saved["image"]).resolve()
        if not image_path.is_relative_to(session_path / "images"):
            raise ValueError("图库只接收会话 images/ 中的原始 RGB")
        records = [item for item in manifest["frames"] if item["image"] == saved["image"]]
        if len(records) != 1 or records[0]["image_sha256"] != saved["image_sha256"]:
            raise ValueError("保存记录与会话清单不一致")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != saved["image_sha256"]:
            raise ValueError("保存后的原始 RGB 校验失败")
        camera_path = session_path / manifest["camera_file"]
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        register_image(
            image_path, library_root=library_root,
            date=saved["received_at_utc"][:10],
            source={"kind": manifest["kind"], "session_manifest": str(manifest_path),
                    "image": saved["image"], "image_sha256": saved["image_sha256"],
                    "received_at_utc": saved["received_at_utc"],
                    "session_id": manifest["session_id"], "batch_id": manifest["batch_id"],
                    "original_split": manifest["split"], "metadata": manifest.get("metadata", {}),
                    "camera_file": str(camera_path), "camera_sha256": manifest["camera_sha256"],
                    "camera_serial": camera["serial"], "device_name": camera["device_name"]},
        )
        print(f"原图已加入统一图库: {library_root}；新图需确认标注后训练。", flush=True)
        return True
    except Exception as exc:
        print(f"原图已保存，但统一图库同步失败，待重新同步: {type(exc).__name__}: {exc}", flush=True)
        return False


def collect(camera, session, interval: float = 0, max_images: int = 0, *,
            library_root: Path | None = None) -> int:
    """Space/S saves the current full frame. A display overlay never enters PNGs."""
    count = 0
    next_save = time.monotonic() + interval
    message = "SPACE/S: save | Q/ESC: finish"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            image, metadata = camera.read()
            preview = image.copy()
            cv2.putText(preview, f"Saved: {count} | {message}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            now = time.monotonic()
            if key in (32, ord("s"), ord("S")) or (interval > 0 and now >= next_save):
                save_metadata = {key: value for key, value in metadata.items() if key != "rgb_settings"}
                saved = session.save(image, **save_metadata)
                if saved is not None:
                    count += 1
                    message = "Saved | SPACE/S: save | Q/ESC: finish"
                    print(f"已保存 {count}: {saved['image']}", flush=True)
                    if not register_saved_rgb(session.path, saved, library_root):
                        message = "Saved; library sync pending | SPACE/S: save | Q/ESC: finish"
                else:
                    message = "Duplicate skipped | SPACE/S: save | Q/ESC: finish"
                    print("相同帧或完全相同图片已跳过", flush=True)
                next_save = now + interval
                if max_images and count >= max_images:
                    break
    finally:
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="D415/D435 原始彩色数据采集：空格/S 保存，Q/ESC 结束")
    parser.add_argument("--config", type=Path, default=ROOT / "config/data_capture_d415.json")
    parser.add_argument("--list-devices", action="store_true", help="只列出型号和序列号，不启动图像流")
    parser.add_argument("--device-model", choices=("D415", "D435"), help="明确选择采集型号，覆盖配置")
    parser.add_argument("--serial", help="完整序列号，覆盖采集配置")
    parser.add_argument("--batch", help="同一拍摄批次的名称；同批图片保持在同一数据分区")
    parser.add_argument("--split", choices=("train", "val", "test", "unknown"), default="train")
    parser.add_argument("--notes", default="", help="记录光照、距离、角度、遮挡等采集条件")
    parser.add_argument("--specimen", help="唯一样件编号")
    parser.add_argument("--printing-batch", help="打印批次编号；同批不跨数据分区")
    parser.add_argument("--geometry-version", help="本次所对应的几何版本")
    parser.add_argument("--source-kind", choices=("printed_connector", "real_connector", "unknown"), default="unknown")
    parser.add_argument("--conditions", type=Path, help="JSON 工况：distance/tilt/image_position 等分层字段")
    parser.add_argument("--output-root", type=Path, help="项目内的数据目录，默认 artifacts/captures")
    library_group = parser.add_mutually_exclusive_group()
    library_group.add_argument("--library", type=Path, default=ROOT / "dataset/vision",
                               help="统一 RGB 图库，默认 dataset/vision；新图待标注")
    library_group.add_argument("--no-library", action="store_true", help="仅保存诊断会话，不加入训练图库")
    parser.add_argument("--interval", type=float, default=0, help="自动保存间隔秒数；默认 0 为手动")
    parser.add_argument("--max-images", type=int, default=0, help="保存指定张数后结束；0 为按键结束")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 0 or args.max_images < 0:
        parser.error("保存间隔和最大图片数必须为非负数")
    if not args.list_devices and (not args.batch or not args.batch.strip()):
        parser.error("采集需要 --batch，例如 --batch d415_session01")
    session = None
    status, error, code = "completed", None, 0
    try:
        if args.list_devices:
            print(json.dumps(list_devices(), ensure_ascii=False, indent=2))
            return 0
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if args.device_model is not None:
            config["device_model"] = args.device_model
        if args.serial is not None:
            config["serial"] = args.serial
        output = args.output_root or Path(config["output_root"])
        output = (output if output.is_absolute() else ROOT / output).resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("采集输出需在项目目录内，以便连接九孔复核页面")
        library_root = None if args.no_library else (
            args.library if args.library.is_absolute() else ROOT / args.library).resolve()
        from .capture_session import CaptureSession
        metadata = {"specimen_id": args.specimen, "printing_batch_id": args.printing_batch,
                    "geometry_version": args.geometry_version, "source_kind": args.source_kind,
                    "training_library_enabled": library_root is not None,
                    "training_library_path": str(library_root) if library_root is not None else None,
                    "conditions": json.loads(args.conditions.read_text(encoding="utf-8-sig")) if args.conditions else {}}
        with CaptureColorCamera(config) as camera:
            session = CaptureSession(output, camera.camera, args.batch, args.split, args.notes, metadata=metadata)
            print(f"采集目录: {session.path}\n空格/S 保存原图，Q/ESC 结束；Ctrl+C 中断。", flush=True)
            collect(camera, session, args.interval, args.max_images, library_root=library_root)
    except KeyboardInterrupt:
        status, code = "interrupted", 130
        print("采集已中断，已保存的数据保留。")
    except Exception as exc:
        status, error, code = "failed", f"{type(exc).__name__}: {exc}", 2
        print(error)
    finally:
        if session is not None:
            try:
                session.finish(status, error=error)
                print(f"批次清单: {session.path / 'manifest.json'}")
            except Exception as exc:
                code = 2
                print(f"批次结束状态保存失败: {exc}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
