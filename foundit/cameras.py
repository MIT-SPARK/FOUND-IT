"""Camera backends for real-time SLAM."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import cv2


class Camera(ABC):
    """Abstract camera that produces BGR uint8 frames."""

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""
        ...

    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Return a BGR uint8 (H, W, 3) frame, or None if unavailable."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release the device."""
        ...


class RealSenseCamera(Camera):

    def start(self) -> None:
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(config)

    def capture(self) -> Optional[np.ndarray]:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self) -> None:
        self._pipeline.stop()


class ZEDCamera(Camera):

    def start(self) -> None:
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        err = self._zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
        self._image_mat = sl.Mat()

    def capture(self) -> Optional[np.ndarray]:
        if self._zed.grab() != self._sl.ERROR_CODE.SUCCESS:
            return None
        self._zed.retrieve_image(self._image_mat, self._sl.VIEW.LEFT)
        bgra = self._image_mat.get_data()  # (H, W, 4) BGRA
        return bgra[:, :, :3].copy()       # (H, W, 3) BGR, contiguous

    def stop(self) -> None:
        self._zed.close()


class LogitechCamera(Camera):
    """Generic UVC webcam (e.g. Logitech) via OpenCV VideoCapture."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {self._device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def capture(self) -> Optional[np.ndarray]:
        # Raise on a failed read (rather than returning None) so the caller's
        # reconnect logic treats a webcam disconnect as a recoverable error.
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Webcam is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam read failed (device may be disconnected)")
        return frame  # already BGR uint8 (H, W, 3)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


BACKENDS = {
    "realsense": RealSenseCamera,
    "zed": ZEDCamera,
    "logitech": LogitechCamera,
}
