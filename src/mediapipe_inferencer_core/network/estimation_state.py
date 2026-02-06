import threading
from dataclasses import dataclass, field


@dataclass
class LandmarkVisualizationSettings:
    pose_enabled: bool = True
    hands_enabled: bool = True
    face_enabled: bool = True


@dataclass
class EstimationState:
    """Thread-safe state management for estimation control."""

    is_running: bool = False
    selected_camera_index: int = 0
    preview_enabled: bool = False
    landmark_visualization: LandmarkVisualizationSettings = field(
        default_factory=LandmarkVisualizationSettings
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Events for signaling state changes
    start_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    stop_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    camera_change_requested: threading.Event = field(default_factory=threading.Event, repr=False)

    def request_start(self) -> bool:
        """Request to start estimation. Returns False if already running."""
        with self._lock:
            if self.is_running:
                return False
            self.start_requested.set()
            return True

    def request_stop(self) -> bool:
        """Request to stop estimation. Returns False if not running."""
        with self._lock:
            if not self.is_running:
                return False
            self.stop_requested.set()
            return True

    def set_running(self, running: bool):
        with self._lock:
            self.is_running = running
            if running:
                self.start_requested.clear()
            else:
                self.stop_requested.clear()

    def set_camera_index(self, index: int):
        with self._lock:
            if self.selected_camera_index != index:
                self.selected_camera_index = index
                self.camera_change_requested.set()

    def get_camera_index(self) -> int:
        with self._lock:
            return self.selected_camera_index

    def acknowledge_camera_change(self):
        self.camera_change_requested.clear()

    def set_preview_enabled(self, enabled: bool):
        with self._lock:
            self.preview_enabled = enabled

    def get_preview_enabled(self) -> bool:
        with self._lock:
            return self.preview_enabled

    def set_landmark_visualization(self, pose: bool, hands: bool, face: bool):
        with self._lock:
            self.landmark_visualization.pose_enabled = pose
            self.landmark_visualization.hands_enabled = hands
            self.landmark_visualization.face_enabled = face

    def get_landmark_visualization(self) -> LandmarkVisualizationSettings:
        with self._lock:
            return LandmarkVisualizationSettings(
                pose_enabled=self.landmark_visualization.pose_enabled,
                hands_enabled=self.landmark_visualization.hands_enabled,
                face_enabled=self.landmark_visualization.face_enabled
            )
