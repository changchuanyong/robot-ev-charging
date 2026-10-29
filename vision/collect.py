"""Collect original D415 color images for training, with per-session provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .contracts import intrinsics_signature, validate_camera


ROOT = Path(__file__).resolve().parents[1]
WINDOW = "D415 data collection"


def list_devices(rs=None) -> list[dict]:
    if rs is None:
        import pyrealsense2 as rs
    context = rs.context()
    return [{"name": device.get_info(rs.camera_info.name),
             "serial": device.get_info(rs.camera_info.serial_number)}
            for device in context.query_devices()]


def validate_config(config: dict) -> None:
    if config.get("device_model") != "D415":
        raise ValueError("数据采集配置的 device_model 必须为 D415")
    if not isinstance(config.get("serial"), str) or not config["serial"].strip():
        raise ValueError("请用 --serial 指定 D415 序列号；--list-devices 可查看设备")
    for name in ("width", "height", "fps", "timeout_ms", "warmup_frames"):
        value = config.get(name)
        minimum = 0 if name == "warmup_frames" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"无效采集参数: {name}")


class D415ColorCamera:
    """Independent data acquisition; never publishes a robot pose."""
    def __init__(self, config: dict):
        validate_config(config)
        self.config = dict(config)
        self.pipeline = None

    def __enter__(self):
        import pyrealsense2 as rs
        serial = self.config["serial"]
        matches = [device for device in list_devices(rs) if device["serial"] == serial]
        if len(matches) != 1:
            raise ValueError("未找到指定序列号的相机，请检查 USB 连接和 --list-devices")
        if matches[0]["name"].split()[-1] != "D415":
            raise ValueError("该序列号不是 D415，数据采集入口拒绝启动其他型号")
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
            if actual_serial != serial or name.split()[-1] != "D415":
                raise ValueError("启动后的相机身份与指定 D415 不一致")
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color.get_intrinsics()
            if (intrinsics.width, intrinsics.height, color.fps(), color.format()) != (
                    self.config["width"], self.config["height"], self.config["fps"], rs.format.bgr8):
                raise ValueError("实际彩色流配置与请求的分辨率、帧率或格式不一致")
            self.camera = {
                "stream": "color", "serial": actual_serial, "device_name": name,
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
            for _ in range(self.config["warmup_frames"]):
                self.read()
        except BaseException:
            self.close()
            raise
        return self

    def read(self) -> tuple[np.ndarray, dict]:
        if self.pipeline is None:
            raise RuntimeError("D415 尚未启动")
        frames = self.pipeline.wait_for_frames(self.config["timeout_ms"])
        received_at = datetime.now(timezone.utc).isoformat()
        color = frames.get_color_frame()
        if not color:
            raise ValueError("D415 没有返回彩色帧")
        image = np.asanyarray(color.get_data()).copy()
        return image, {"frame_number": color.get_frame_number(),
                       "device_timestamp_ms": color.get_timestamp(),
                       "timestamp_domain": color.get_frame_timestamp_domain().name,
                       "received_at_utc": received_at}

    def close(self):
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

    def __exit__(self, *_):
        self.close()


def collect(camera, session, interval: float = 0, max_images: int = 0) -> int:
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
                saved = session.save(image, **metadata)
                if saved is not None:
                    count += 1
                    message = "Saved | SPACE/S: save | Q/ESC: finish"
                    print(f"已保存 {count}: {saved['image']}", flush=True)
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
    parser = argparse.ArgumentParser(description="D415 原始彩色数据采集：空格/S 保存，Q/ESC 结束")
    parser.add_argument("--config", type=Path, default=ROOT / "config/data_capture_d415.json")
    parser.add_argument("--list-devices", action="store_true", help="只列出型号和序列号，不启动图像流")
    parser.add_argument("--serial", help="D415 完整序列号，覆盖采集配置")
    parser.add_argument("--batch", help="同一拍摄批次的名称；同批图片保持在同一数据分区")
    parser.add_argument("--split", choices=("train", "val", "test", "unknown"), default="train")
    parser.add_argument("--notes", default="", help="记录光照、距离、角度、遮挡等采集条件")
    parser.add_argument("--output-root", type=Path, help="项目内的数据目录，默认 artifacts/captures")
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
        if args.serial is not None:
            config["serial"] = args.serial
        output = args.output_root or Path(config["output_root"])
        output = (output if output.is_absolute() else ROOT / output).resolve()
        if not output.is_relative_to(ROOT):
            raise ValueError("采集输出需在项目目录内，以便连接九孔复核页面")
        from .capture_session import CaptureSession
        with D415ColorCamera(config) as camera:
            session = CaptureSession(output, camera.camera, args.batch, args.split, args.notes)
            print(f"采集目录: {session.path}\n空格/S 保存原图，Q/ESC 结束；Ctrl+C 中断。", flush=True)
            collect(camera, session, args.interval, args.max_images)
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
