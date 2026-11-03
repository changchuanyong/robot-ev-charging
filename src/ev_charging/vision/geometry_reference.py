"""Versioned aperture geometry and explicit CAD sections; never infer print dimensions."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


LABELS = ("S_neg", "CC2", "S_pos", "CC1", "DC_neg", "DC_pos", "A_neg", "PE", "A_pos")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty_{name}_required")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256_required")


def load_geometry_reference(path: str | Path) -> dict:
    """Load finite named millimetre points; nominal models remain diagnostic only.

    Provenance hashes are recorded, not evidence of metrological accuracy. A
    measured_print record needs an accountable measurement record and explicit
    aperture sections. No hardware acceptance is granted by this loader.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1 or data.get("units") != "mm" or data.get("frame") != "target":
        raise ValueError("geometry_requires_schema1_target_frame_mm")
    for key in ("version", "feature_definition"):
        _text(data.get(key), key)
    if data.get("status") not in {"nominal_reference", "cad_extracted", "measured_print"}:
        raise ValueError("invalid_geometry_status")
    if not isinstance(data.get("validated"), bool):
        raise ValueError("explicit_geometry_validation_status_required")
    if data["status"] != "measured_print" and data["validated"]:
        raise ValueError("nominal_or_CAD_is_not_validated_print_geometry")
    points, sections = data.get("points_mm"), data.get("sections")
    if not isinstance(points, dict) or set(points) != set(LABELS):
        raise ValueError("all_nine_named_geometry_points_required")
    if not isinstance(sections, dict) or set(sections) != set(LABELS):
        raise ValueError("all_nine_aperture_sections_required")
    for label in LABELS:
        values = points[label]
        if (not isinstance(values, list) or len(values) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
                or not np.isfinite(values).all()):
            raise ValueError(f"finite_xyz_mm_required:{label}")
        _text(sections[label], f"section_{label}")
    coordinates = np.asarray(list(points.values()), dtype=float)
    if len(np.unique(coordinates, axis=0)) != len(LABELS) or np.linalg.matrix_rank(coordinates-coordinates.mean(0)) < 2:
        raise ValueError("degenerate_geometry_points")
    provenance = data.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("geometry_provenance_required")
    for item in provenance:
        for key in ("path", "kind"):
            _text(item.get(key), key)
        _digest(item.get("sha256"))
    if data["status"] == "measured_print":
        for key in ("specimen_id", "printing_batch_id"):
            _text(data.get(key), key)
        measurement = data.get("measurement")
        if not isinstance(measurement, dict):
            raise ValueError("print_measurement_record_required")
        for key in ("operator", "measured_at", "instrument", "evidence_path"):
            _text(measurement.get(key), key)
        if datetime.fromisoformat(measurement["measured_at"].replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("measurement_timezone_required")
        _digest(measurement.get("evidence_sha256"))
    return {**data, "source_path": str(path.resolve()), "source_sha256": sha256(path)}


def read_binary_stl(path: Path) -> np.ndarray:
    blob = Path(path).read_bytes()
    if len(blob) < 84:
        raise ValueError("binary_stl_header_missing")
    count = struct.unpack_from("<I", blob, 80)[0]
    if count < 1 or len(blob) != 84+50*count:
        raise ValueError("requires_binary_stl_with_exact_triangle_count")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(blob, dtype=dtype, count=count, offset=84)["vertices"].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError("nonfinite_mesh")
    return triangles


def section_circles(triangles: np.ndarray, axis: int, level: float, tolerance=1e-5) -> list[dict]:
    """Intersect mesh with an explicit plane and fit closed loop candidates.

    The output contains ALL closed loops, including outer outlines and steps.
    Circle residual and angular coverage help inspect candidates; labels and
    physical aperture identity must be supplied by the CAD/print reviewer.
    Planes through mesh vertices are rejected to avoid ambiguous coplanar edges.
    """
    if axis not in (0, 1, 2) or not np.isfinite([level, tolerance]).all() or tolerance <= 0:
        raise ValueError("invalid_section_plane")
    if np.any(np.abs(triangles[:, :, axis]-level) < tolerance):
        raise ValueError("section_hits_vertices_choose_interior_plane")
    remaining = [i for i in range(3) if i != axis]
    graph, coordinates = {}, {}
    for tri in triangles:
        side = tri[:, axis]-level
        if np.all(side > 0) or np.all(side < 0):
            continue
        crossing = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if side[a]*side[b] < 0:
                p = tri[a]+side[a]/(side[a]-side[b])*(tri[b]-tri[a])
                crossing.append(p)
        if len(crossing) != 2:
            raise ValueError("ambiguous_triangle_section")
        keys = [tuple(np.rint(p/tolerance).astype(np.int64)) for p in crossing]
        if keys[0] == keys[1]:
            continue
        for i in (0, 1):
            coordinates[keys[i]] = crossing[i]
            graph.setdefault(keys[i], set()).add(keys[1-i])
    remaining_nodes = set(graph)
    loops = []
    while remaining_nodes:
        stack = [min(remaining_nodes)]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(graph[node]-component)
        remaining_nodes -= component
        if len(component) < 6 or any(len(graph[p]) != 2 for p in component):
            continue
        xyz = np.array([coordinates[p] for p in sorted(component)])
        xy = xyz[:, remaining]
        centered = xy-xy.mean(0)
        solution = np.linalg.lstsq(np.c_[2*centered, np.ones(len(xy))], np.sum(centered**2, axis=1), rcond=None)[0]
        center = solution[:2]+xy.mean(0)
        radii = np.linalg.norm(xy-center, axis=1)
        radius = float(np.mean(radii))
        angles = np.sort(np.arctan2(xy[:, 1]-center[1], xy[:, 0]-center[0]))
        largest_gap = float(np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])))
        center3 = np.zeros(3)
        center3[axis], center3[remaining] = level, center
        loops.append({"center_cad": center3.tolist(), "diameter": 2*radius,
                      "radial_rms": float(np.sqrt(np.mean((radii-radius)**2))),
                      "radial_max": float(np.max(np.abs(radii-radius))),
                      "largest_angular_gap_rad": largest_gap, "vertex_count": len(xy)})
    loops.sort(key=lambda item: tuple(item["center_cad"])+tuple([item["diameter"]]))
    return [{"candidate_id": i+1, **row} for i, row in enumerate(loops)]


def audit_sources(directory: Path) -> dict:
    rows = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in {".stl", ".sldprt", ".3mf", ".pdf"}:
            continue
        row = {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size,
               "kind": path.suffix.lower()[1:]}
        if path.suffix.lower() == ".stl":
            triangles = read_binary_stl(path)
            vertices = triangles.reshape(-1, 3)
            row.update(triangles=len(triangles), units="unspecified_in_STL",
                       bounds_min=vertices.min(0).tolist(), bounds_max=vertices.max(0).tolist(),
                       dimensions=(vertices.max(0)-vertices.min(0)).tolist())
        elif path.suffix.lower() == ".3mf":
            with zipfile.ZipFile(path) as archive:
                row["declared_model_units"] = {name: ET.fromstring(archive.read(name)).get("unit", "millimeter")
                                               for name in archive.namelist() if name.endswith(".model")}
        rows.append(row)
    return {"schema_version": 1, "kind": "CAD_source_inventory", "validated_print_geometry": False,
            "sources": rows, "pending": ["select aperture cross-sections and CAD-to-target frame",
                                            "measure actual print dimensions, steps, clearances and insertion depth"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("geometry", type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--cad-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    section = commands.add_parser("section")
    section.add_argument("--mesh", type=Path, required=True)
    section.add_argument("--axis", choices=["x", "y", "z"], required=True)
    section.add_argument("--level", type=float, required=True)
    section.add_argument("--unit", choices=["mm"], required=True, help="Explicit CAD export unit; STL has no units")
    section.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        data = load_geometry_reference(args.geometry)
        print(json.dumps({key: data[key] for key in ("version", "status", "validated", "source_sha256")}, ensure_ascii=False))
        return
    if args.command == "audit":
        data = audit_sources(args.cad_dir)
    else:
        data = {"schema_version": 1, "kind": "unassigned_CAD_section_candidates", "validated": False,
                "mesh_path": str(args.mesh.resolve()), "mesh_sha256": sha256(args.mesh),
                "axis": args.axis, "level": args.level, "units": args.unit,
                "candidates": section_circles(read_binary_stl(args.mesh), "xyz".index(args.axis), args.level),
                "warning": "CAD section candidates are not named apertures or measured print geometry"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
