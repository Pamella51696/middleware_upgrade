"""Load YAML config and keep per-parameter provenance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

CAMERA_ORDER = ("FRONT", "LEFT", "RIGHT", "REAR")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_all(config_dir: Path | None = None) -> Dict[str, Any]:
    root = Path(config_dir) if config_dir else CONFIG_DIR
    cameras = load_yaml(root / "cameras.yaml")
    projection = load_yaml(root / "projection.yaml")
    stitching = load_yaml(root / "stitching.yaml")
    return {
        "cameras": cameras,
        "projection": projection,
        "stitching": stitching,
        "config_dir": str(root),
    }


def camera_ids(cfg: Dict[str, Any]) -> List[str]:
    cams = cfg["cameras"]["cameras"]
    return [cid for cid in CAMERA_ORDER if cid in cams]


def camera_cfg(cfg: Dict[str, Any], camera_id: str) -> Dict[str, Any]:
    return cfg["cameras"]["cameras"][camera_id]


def save_cameras(cfg: Dict[str, Any], path: Path | None = None) -> None:
    """Persist alignment / intrinsic edits without dropping provenance fields."""
    root = Path(path) if path else Path(cfg["config_dir"]) / "cameras.yaml"
    dump = deepcopy(cfg["cameras"])
    with root.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dump, f, sort_keys=False, default_flow_style=False)


def merge_camera_overrides(cfg: Dict[str, Any], camera_id: str, overrides: Dict[str, Any]) -> None:
    """Shallow-merge slider updates into a live config (debug UI)."""
    node = camera_cfg(cfg, camera_id)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(node.get(key), dict):
            node[key].update(value)
        else:
            node[key] = value
