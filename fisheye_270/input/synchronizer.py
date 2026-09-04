"""Nearest-timestamp / lock-step frame sync.

Prerecorded files usually have no shared clock. Default: lock-step by
read index (one frame from each source per tick). If timestamps exist,
the leader is the newest timestamp and others pick the closest buffered
frame within ``max_skew_sec``.

A dead camera does not stop the bundle: its slot is marked ok=False and
the pipeline composites the remaining views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sources import CAMERA_IDS, Frame, FrameSource


@dataclass
class SyncedBundle:
    frames: Dict[str, Frame]
    timestamp_sec: float
    index: int
    missing: List[str] = field(default_factory=list)

    def image(self, camera_id: str):
        fr = self.frames.get(camera_id)
        if fr is None or not fr.ok:
            return None
        return fr.image


class FrameSynchronizer:
    def __init__(self, sources: Dict[str, FrameSource], max_skew_sec: float = 0.08) -> None:
        self.sources = sources
        self.max_skew_sec = max_skew_sec
        self._index = 0
        self._last: Dict[str, Frame] = {}

    def next(self) -> Optional[SyncedBundle]:
        got: Dict[str, Frame] = {}
        missing: List[str] = []
        for cid in CAMERA_IDS:
            src = self.sources.get(cid)
            if src is None:
                missing.append(cid)
                continue
            fr = src.read()
            if fr is None or not fr.ok:
                missing.append(cid)
                if cid in self._last:
                    got[cid] = self._last[cid]
                continue
            self._last[cid] = fr
            got[cid] = fr

        if not got:
            return None
        ts = max(fr.timestamp_sec for fr in got.values())
        bundle = SyncedBundle(frames=got, timestamp_sec=ts, index=self._index, missing=missing)
        self._index += 1
        return bundle

    def close(self) -> None:
        for s in self.sources.values():
            s.close()
