"""Thesis equations (3-16)--(3-18): signed margins of a bounded wrench domain.

This is an engineering margin, not a probability of jamming. Calibration
readiness is separate from mathematical validity, allowing explicit simulation
domains to be tested without representing them as measured hardware limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RiskResult:
    valid: bool
    values: np.ndarray
    outside: bool
    reason: str


class EngineeringRisk:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.version = str(config.get("version", ""))
        self.calibrated = config.get("calibrated", False) is True
        self.scale = np.asarray(config["scale"], dtype=float)
        self.A = np.asarray(config["A"], dtype=float)
        self.b = np.asarray(config["b"], dtype=float)
        self.reference = np.asarray(config["reference"], dtype=float)
        if not self.version:
            raise ValueError("engineering domain needs a version")
        if (self.scale.shape != (6,) or not np.all(np.isfinite(self.scale))
                or np.any(self.scale <= 0)):
            raise ValueError("risk scale must contain six positive finite values")
        if (self.A.ndim != 2 or self.A.shape[1] != 6 or self.A.shape[0] < 12
                or self.b.shape != (self.A.shape[0],)
                or not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.b))):
            raise ValueError("domain needs finite A[m,6] and b[m], including signed axis bounds")
        if self.reference.shape != (6,) or not np.all(np.isfinite(self.reference)):
            raise ValueError("domain reference must be a finite normalized six-vector")
        if np.any(np.linalg.norm(self.A, axis=1) <= 1e-12):
            raise ValueError("zero domain rows are not allowed")
        # Explicit positive/negative axis bounds guarantee a bounded domain
        # without an optional linear-program solver. Extra coupling rows are
        # allowed, but cannot replace the documented independent load limits.
        nonzero = np.abs(self.A) > 1e-12
        for axis in range(6):
            axis_rows = np.sum(nonzero, axis=1) == 1
            column = self.A[axis_rows, axis]
            if not (np.any(column > 1e-12) and np.any(column < -1e-12)):
                raise ValueError(f"missing explicit positive/negative bound for axis {axis}")
        slack = self.b - self.A @ self.reference
        if np.any(slack <= 1e-10):
            raise ValueError("domain reference must be strictly inside every constraint")
        self._reference_margins = self._margins(slack)

    def _margins(self, slack: np.ndarray) -> np.ndarray:
        margins = np.empty((2, 6))
        for side, sign in enumerate((-1.0, 1.0)):
            for axis in range(6):
                coefficient = sign * self.A[:, axis]
                active = coefficient > 1e-12
                margins[side, axis] = np.min(slack[active] / coefficient[active])
        return margins

    def evaluate(self, wrench) -> RiskResult:
        try:
            wrench = np.asarray(wrench, dtype=float)
        except (TypeError, ValueError):
            return RiskResult(False, np.ones(6), False, "invalid_wrench")
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            return RiskResult(False, np.ones(6), False, "invalid_wrench")
        normalized = wrench / self.scale
        slack = self.b - self.A @ normalized
        if not np.all(np.isfinite(slack)):
            return RiskResult(False, np.ones(6), False, "invalid_wrench")
        if np.any(slack < -1e-10):
            return RiskResult(False, np.ones(6), True, "outside_engineering_domain")
        margins = self._margins(np.maximum(slack, 0.0))
        values = np.max(np.clip(1.0 - margins / self._reference_margins, 0.0, 1.0), axis=0)
        return RiskResult(True, values, False, "ok")


CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
UNITS = ["N", "N", "N", "Nm", "Nm", "Nm"]
IDENTITIES = ("coordinate_version", "specimen_version", "tool_version", "wrench_calibration_version")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity/version")
    return value


def _vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain six finite values")
    return vector


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_domain(spec: dict, base_dir: Path = Path(".")) -> dict:
    """Compile documented signed limits into a reproducible engineering risk domain.

    Usage: python -m control.risk --spec spec.json --output domain.json
           [--records independent_validation.jsonl ...]

    Spec schema (all limits must come from independent engineering acceptance):
      schema_version: 1; version; coordinate_version; specimen_version; tool_version;
      wrench_calibration_version; frame: task; moment_origin: plug;
      units: [N,N,N,Nm,Nm,Nm]; reference_wrench: six SI values; reference_source: id;
      sources: [{id, version, path, sha256(optional)}];
      limits: {Fx: {lower,upper,lower_source,upper_source}, ... Fy,Fz,Mx,My,Mz};
      scale: six positive SI scales (optional; otherwise max(abs(lower),abs(upper)));
      coupled: [{a: six coefficients, b, space: normalized, source: id, version}].

    Sources are local evidence files, relative to the spec file. Each is hashed.
    `a @ (w / scale) <= b` defines an optional coupling; it never replaces axis
    bounds. No observations are fitted, and geometric dimensions do not imply safe
    loads. Output is ALWAYS calibrated=false; formal manual acceptance is separate.

    Validation JSONL starts with a manifest whose `risk_validation` object (also
    accepted under manifest.config) declares split=validation, domain_version and
    the coordinate/specimen/tool/wrench versions, frame, moment_origin and units
    above. Cycle records use decision.details.wrench (current, unfiltered, at tool
    in task axes). Sample records must explicitly declare wrench_frame=task and
    metadata.moment_origin=tool|plug; raw sensor records are refused, not converted.
    """
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("spec schema_version must equal 1")
    if spec.get("units") != UNITS or spec.get("frame") != "task" or spec.get("moment_origin") != "plug":
        raise ValueError("domain requires task-frame plug-origin N/Nm wrench units")
    version = _text(spec.get("version"), "version")
    identities = {key: _text(spec.get(key), key) for key in IDENTITIES}
    sources = {}
    for source in spec["sources"]:
        if not isinstance(source, dict):
            raise ValueError("each evidence source must be an object")
        identifier = _text(source.get("id"), "source.id")
        if identifier in sources:
            raise ValueError("duplicate source identity")
        source_version = _text(source.get("version"), "source.version")
        path = (base_dir / source["path"]).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"nonempty source evidence file required: {path}")
        digest = _digest_file(path)
        if source.get("sha256") is not None and source["sha256"] != digest:
            raise ValueError(f"source evidence hash mismatch: {identifier}")
        sources[identifier] = {"id": identifier, "version": source_version,
                               "path": str(path), "sha256": digest}
    def source_id(value):
        if value not in sources:
            raise ValueError(f"missing source evidence: {value}")
        return value
    reference_source = source_id(spec["reference_source"])
    lower, upper, lower_sources, upper_sources = [], [], [], []
    if set(spec["limits"]) != set(CHANNELS):
        raise ValueError("limits must explicitly specify Fx,Fy,Fz,Mx,My,Mz")
    for channel in CHANNELS:
        row = spec["limits"][channel]
        lo, hi = float(row["lower"]), float(row["upper"])
        if not np.isfinite([lo, hi]).all() or not lo < 0 < hi:
            raise ValueError(f"{channel} needs explicit negative lower and positive upper bounds")
        lower.append(lo)
        upper.append(hi)
        lower_sources.append(source_id(row["lower_source"]))
        upper_sources.append(source_id(row["upper_source"]))
    lower, upper = np.asarray(lower), np.asarray(upper)
    scale = _vector(spec.get("scale", np.maximum(-lower, upper)), "scale")
    if np.any(scale <= 0):
        raise ValueError("scale must be positive")
    matrix = list(np.eye(6)) + list(-np.eye(6))
    bounds = list(upper / scale) + list(-lower / scale)
    row_sources = [{"constraint": f"{channel}_upper", "source": source}
                   for channel, source in zip(CHANNELS, upper_sources)]
    row_sources += [{"constraint": f"{channel}_lower", "source": source}
                    for channel, source in zip(CHANNELS, lower_sources)]
    for index, row in enumerate(spec.get("coupled", [])):
        if row.get("space") != "normalized":
            raise ValueError("coupled coefficients must explicitly use normalized wrench space")
        coefficients = _vector(row["a"], "coupled.a")
        bound = float(row["b"])
        if not np.isfinite(bound) or np.linalg.norm(coefficients) <= 1e-12:
            raise ValueError("invalid coupling row")
        matrix.append(coefficients)
        bounds.append(bound)
        row_sources.append({"constraint": f"coupled_{index}", "source": source_id(row["source"]),
                            "version": _text(row.get("version"), "coupled.version")})
    reference = _vector(spec["reference_wrench"], "reference_wrench") / scale
    domain = {"version": version, "scale": scale.tolist(), "A": np.asarray(matrix).tolist(),
              "b": bounds, "reference": reference.tolist(), "calibrated": False}
    EngineeringRisk(domain)  # Includes strict interior reference and boundedness checks.
    domain_hash = hashlib.sha256(json.dumps(domain, sort_keys=True, allow_nan=False).encode()).hexdigest()
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
    domain["provenance"] = {**identities, "frame": "task", "moment_origin": "plug", "units": UNITS,
                            "spec_sha256": spec_hash, "domain_sha256": domain_hash,
                            "sources": list(sources.values()), "constraint_sources": row_sources,
                            "reference_source": reference_source}
    domain["acceptance"] = {"status": "manual_review_required", "calibrated_by_builder": False,
                            "note": "Hashing evidence and reporting log coverage do not approve hardware limits."}
    return domain


def validate_records(domain: dict, paths: list[Path]) -> dict:
    """Report inclusion in the fixed domain; never expand or fit its boundaries."""
    risk = EngineeringRisk(domain)
    provenance = domain["provenance"]
    reports, seen_hashes, seen_tasks = [], set(), set()
    source_hashes = {item["sha256"] for item in provenance["sources"]}
    for path in paths:
        path = Path(path)
        digest = _digest_file(path)
        if digest in seen_hashes:
            raise ValueError("duplicate validation log contents")
        if digest in source_hashes:
            raise ValueError("independent validation cannot reuse boundary-source evidence")
        seen_hashes.add(digest)
        with path.open(encoding="utf-8-sig") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows or not all(isinstance(row, dict) for row in rows) or rows[0].get("kind") != "manifest":
            raise ValueError("validation log must begin with a manifest")
        manifest = rows[0]
        task_id = _text(manifest.get("task_id"), "validation.task_id")
        if task_id in seen_tasks:
            raise ValueError("duplicate task identity in validation logs")
        seen_tasks.add(task_id)
        source = manifest.get("source")
        if source not in {"hardware", "simulation"}:
            raise ValueError("validation source must explicitly be hardware or simulation")
        config = manifest.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError("manifest config must be an object")
        identity = manifest.get("risk_validation", config.get("risk_validation", {}))
        expected = {key: provenance[key] for key in (*IDENTITIES, "frame", "moment_origin", "units")}
        expected.update(split="validation", domain_version=domain["version"])
        if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("validation partition/domain/coordinate/specimen/tool identity mismatch")
        inside = outside = invalid = ignored = 0
        minimum = np.full(6, np.inf)
        maximum = np.full(6, -np.inf)
        for row in rows[1:]:
            if row.get("task_id") != task_id:
                raise ValueError("record task identity differs from manifest")
            if row.get("kind") == "cycle":
                if row.get("sample", {}).get("valid") is not True:
                    invalid += 1
                    continue
                if row.get("decision", {}).get("state") == "VISION":
                    ignored += 1  # Task axes have not been frozen yet.
                    continue
                wrench = row.get("decision", {}).get("details", {}).get("wrench")
            elif row.get("kind") == "sample":
                sample = row.get("sample", {})
                if sample.get("wrench_frame") != "task" or sample.get("metadata", {}).get("moment_origin") not in {"tool", "plug"}:
                    raise ValueError("sample validation requires explicit task-axis plug-origin wrench; raw data is not converted")
                if sample.get("valid") is not True:
                    invalid += 1
                    continue
                wrench = sample.get("wrench")
            else:
                ignored += 1
                continue
            result = risk.evaluate(wrench)
            if result.outside:
                outside += 1
            elif result.valid:
                inside += 1
            else:
                invalid += 1
                continue
            vector = np.asarray(wrench, dtype=float)
            minimum, maximum = np.minimum(minimum, vector), np.maximum(maximum, vector)
        count = inside + outside
        if count == 0:
            raise ValueError("validation log contains no evaluable task-frame wrench samples")
        reports.append({"path": str(path.resolve()), "sha256": digest, "task_id": task_id, "source": source,
                        "evaluable_samples": count, "inside_samples": inside, "outside_samples": outside,
                        "invalid_samples": invalid, "ignored_records": ignored,
                        "domain_inclusion_fraction": inside/count, "outside_fraction": outside/count,
                        "observed_minimum": minimum.tolist(), "observed_maximum": maximum.tolist()})
    return {"logs": reports, "boundaries_refitted": False,
            "note": "Sample inclusion is not task success, hazard probability or proof of safety; simulation is labeled separately."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records", nargs="+", type=Path, default=[])
    args = parser.parse_args(argv)
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))
        domain = build_domain(spec, args.spec.resolve().parent)
        domain["validation"] = validate_records(domain, args.records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(domain, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "calibrated": False,
                          "constraints": len(domain["b"]), "validation_logs": len(args.records)}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Risk-domain build rejected: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
