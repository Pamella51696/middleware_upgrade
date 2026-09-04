from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


CAMERA_IDS = ("FRONT", "LEFT", "RIGHT", "REAR")


@dataclass
class Frame:
    camera_id: str
    image: np.ndarray
    timestamp_sec: float
    index: int
    ok: bool = True


class FrameSource:
    def read(self) -> Optional[Frame]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def is_open(self) -> bool:
        return True


class VideoReader(FrameSource):
    def __init__(self, camera_id: str, path: str, loop: bool = True) -> None:
        self.camera_id = camera_id
        self.path = path
        self.loop = loop
        self._cap = cv2.VideoCapture(path)
        self._index = 0
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        if not self._cap.isOpened():
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Optional[Frame]:
        if not self.is_open:
            return Frame(self.camera_id, np.zeros((1, 1, 3), np.uint8), 0.0, self._index, ok=False)
        ok, img = self._cap.read()
        if not ok or img is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
            if not ok or img is None:
                return Frame(self.camera_id, np.zeros((1, 1, 3), np.uint8), 0.0, self._index, ok=False)
        ts = self._index / self._fps
        fr = Frame(self.camera_id, img, ts, self._index, ok=True)
        self._index += 1
        return fr

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSequence(FrameSource):
    def __init__(self, camera_id: str, paths: list[str], fps: float = 15.0, loop: bool = True) -> None:
        self.camera_id = camera_id
        self.paths = list(paths)
        self.fps = fps
        self.loop = loop
        self._i = 0

    def read(self) -> Optional[Frame]:
        if not self.paths:
            return Frame(self.camera_id, np.zeros((1, 1, 3), np.uint8), 0.0, 0, ok=False)
        if self._i >= len(self.paths):
            if not self.loop:
                return Frame(self.camera_id, np.zeros((1, 1, 3), np.uint8), 0.0, self._i, ok=False)
            self._i = 0
        img = cv2.imread(self.paths[self._i], cv2.IMREAD_COLOR)
        ok = img is not None
        if not ok:
            img = np.zeros((1, 1, 3), np.uint8)
        fr = Frame(self.camera_id, img, self._i / self.fps, self._i, ok=ok)
        self._i += 1
        return fr


class SyntheticSource(FrameSource):
    """In-memory repeating frames (tests / demo without files)."""

    def __init__(self, camera_id: str, frames: list[np.ndarray], fps: float = 15.0) -> None:
        self.camera_id = camera_id
        self.frames = frames
        self.fps = fps
        self._i = 0

    def read(self) -> Optional[Frame]:
        img = self.frames[self._i % len(self.frames)]
        fr = Frame(self.camera_id, img, self._i / self.fps, self._i, ok=True)
        self._i += 1
        return fr
