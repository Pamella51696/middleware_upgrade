from pathlib import Path
from typing import Any, Dict

from .sources import FrameSource, ImageSequence, SyntheticSource, VideoReader
from .synchronizer import FrameSynchronizer, SyncedBundle

__all__ = [
    "FrameSource",
    "FrameSynchronizer",
    "ImageSequence",
    "SyncedBundle",
    "SyntheticSource",
    "VideoReader",
    "open_configured_sources",
]


def open_configured_sources(cfg: Dict[str, Any], loop: bool = True) -> Dict[str, FrameSource]:
    cams = cfg["cameras"]["cameras"]
    sources: Dict[str, FrameSource] = {}
    for cid, cam in cams.items():
        if not cam.get("enabled", True):
            continue
        src = cam.get("source") or {}
        kind = str(src.get("type") or "video")
        path = src.get("path")
        if kind == "video" and path:
            sources[cid] = VideoReader(cid, str(path), loop=loop)
        elif kind == "images" and path:
            files = sorted(Path(path).glob("*"))
            sources[cid] = ImageSequence(cid, [str(p) for p in files], loop=loop)
        # synthetic is injected by the CLI / tests
    return sources
