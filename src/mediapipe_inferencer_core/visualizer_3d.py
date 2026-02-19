import open3d as o3d
import numpy as np
from mediapipe_inferencer_core.data_class import LandmarkResult, HandResult

# MediaPipe Pose landmark indices
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

# MediaPipe Hand landmark indices
HAND_WRIST = 0

# MediaPipe Pose connections (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),      # Right eye
    (0, 4), (4, 5), (5, 6), (6, 8),      # Left eye
    (9, 10),                              # Mouth
    (11, 12),                             # Shoulders
    (11, 13), (13, 15),                   # Left arm
    (12, 14), (14, 16),                   # Right arm
    (15, 17), (15, 19), (15, 21),         # Left hand
    (16, 18), (16, 20), (16, 22),         # Right hand
    (17, 19), (18, 20),                   # Hand connections
    (11, 23), (12, 24), (23, 24),         # Torso
    (23, 25), (25, 27),                   # Left leg
    (24, 26), (26, 28),                   # Right leg
    (27, 29), (27, 31), (29, 31),         # Left foot
    (28, 30), (28, 32), (30, 32),         # Right foot
]

# MediaPipe Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17),            # Palm
]


def create_ground_grid(size: float = 2.0, divisions: int = 20, y: float = 0.0) -> o3d.geometry.LineSet:
    """Create a grid on the XZ plane (ground)."""
    lines = []
    points = []
    half = size / 2
    step = size / divisions

    # Lines along X axis
    for i in range(divisions + 1):
        z = -half + i * step
        points.append([-half, y, z])
        points.append([half, y, z])
        lines.append([len(points) - 2, len(points) - 1])

    # Lines along Z axis
    for i in range(divisions + 1):
        x = -half + i * step
        points.append([x, y, -half])
        points.append([x, y, half])
        lines.append([len(points) - 2, len(points) - 1])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
    return grid


class Pose3DVisualizer:
    """Real-time 3D visualizer for MediaPipe pose and hand landmarks using Open3D.

    Controls:
        W/S: Move forward/backward
        A/D: Move left/right
        Q/E: Move down/up
        R: Reset camera position
    """

    def __init__(self, window_name: str = "Pose 3D", width: int = 800, height: int = 600):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name, width=width, height=height)

        # Camera state
        self.camera_pos = np.array([0.0, 1.0, -5.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.move_speed = 0.3

        # Register key callbacks (WASD + QE)
        self.vis.register_key_callback(ord('W'), self._move_forward)
        self.vis.register_key_callback(ord('S'), self._move_backward)
        self.vis.register_key_callback(ord('A'), self._move_left)
        self.vis.register_key_callback(ord('D'), self._move_right)
        self.vis.register_key_callback(ord('Q'), self._move_down)
        self.vis.register_key_callback(ord('E'), self._move_up)
        self.vis.register_key_callback(ord('R'), self._reset_camera)

        # Pose geometry
        self.pose_pcd = o3d.geometry.PointCloud()
        self.pose_lines = o3d.geometry.LineSet()

        # Hand geometry (left and right)
        self.left_hand_pcd = o3d.geometry.PointCloud()
        self.left_hand_lines = o3d.geometry.LineSet()
        self.right_hand_pcd = o3d.geometry.PointCloud()
        self.right_hand_lines = o3d.geometry.LineSet()

        # Add all geometries to visualizer
        self.vis.add_geometry(self.pose_pcd)
        self.vis.add_geometry(self.pose_lines)
        self.vis.add_geometry(self.left_hand_pcd)
        self.vis.add_geometry(self.left_hand_lines)
        self.vis.add_geometry(self.right_hand_pcd)
        self.vis.add_geometry(self.right_hand_lines)

        # Add ground grid (3m x 3m, 15cm per cell)
        ground_grid = create_ground_grid(size=3.0, divisions=20, y=0.0)
        self.vis.add_geometry(ground_grid)

        # Initial render to setup view
        self.vis.poll_events()
        self.vis.update_renderer()
        self._apply_camera()

    def _move_forward(self, vis):
        self.camera_pos[2] += self.move_speed
        self.camera_target[2] += self.move_speed
        self._apply_camera()
        return False

    def _move_backward(self, vis):
        self.camera_pos[2] -= self.move_speed
        self.camera_target[2] -= self.move_speed
        self._apply_camera()
        return False

    def _move_left(self, vis):
        self.camera_pos[0] -= self.move_speed
        self.camera_target[0] -= self.move_speed
        self._apply_camera()
        return False

    def _move_right(self, vis):
        self.camera_pos[0] += self.move_speed
        self.camera_target[0] += self.move_speed
        self._apply_camera()
        return False

    def _move_up(self, vis):
        self.camera_pos[1] += self.move_speed
        self.camera_target[1] += self.move_speed
        self._apply_camera()
        return False

    def _move_down(self, vis):
        self.camera_pos[1] -= self.move_speed
        self.camera_target[1] -= self.move_speed
        self._apply_camera()
        return False

    def _reset_camera(self, vis):
        self.camera_pos = np.array([0.0, 1.0, -5.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self._apply_camera()
        return False

    def _apply_camera(self):
        """Apply current camera position to visualizer."""
        ctr = self.vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()

        up = np.array([0, -1, 0])

        # Calculate camera rotation matrix (look-at)
        forward = self.camera_target - self.camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        actual_up = np.cross(forward, right)

        # Build extrinsic matrix (world to camera)
        extrinsic = np.eye(4)
        extrinsic[:3, 0] = right
        extrinsic[:3, 1] = actual_up
        extrinsic[:3, 2] = forward
        extrinsic[:3, 3] = self.camera_pos

        # Invert to get camera extrinsic (camera to world -> world to camera)
        extrinsic = np.linalg.inv(extrinsic)

        param.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

    def _update_point_cloud(self, pcd: o3d.geometry.PointCloud,
                            landmarks: np.ndarray, color: list):
        """Update point cloud geometry with new landmarks."""
        if landmarks is None or len(landmarks) == 0:
            pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            return

        # landmarks shape: (N, 4) -> [x, y, z, confidence]
        # Flip X axis: remove mirror effect (MediaPipe X+ is subject's right)
        # Flip Y axis: MediaPipe is Y-down, Open3D is Y-up
        points = landmarks[:, :3].copy()
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)

    def _update_line_set(self, lines: o3d.geometry.LineSet,
                         landmarks: np.ndarray, connections: list, color: list):
        """Update line set geometry with new landmarks and connections."""
        if landmarks is None or len(landmarks) == 0:
            lines.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            lines.lines = o3d.utility.Vector2iVector([])
            return

        # Flip X axis: remove mirror effect (MediaPipe X+ is subject's right)
        # Flip Y axis: MediaPipe is Y-down, Open3D is Y-up
        points = landmarks[:, :3].copy()
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]
        # Filter valid connections
        valid_connections = [
            conn for conn in connections
            if conn[0] < len(points) and conn[1] < len(points)
        ]

        lines.points = o3d.utility.Vector3dVector(points)
        lines.lines = o3d.utility.Vector2iVector(valid_connections)
        lines.paint_uniform_color(color)

    def _align_hand_to_pose_wrist(self, hand_landmarks: np.ndarray,
                                   pose_wrist: np.ndarray) -> np.ndarray:
        """Align hand landmarks to pose wrist position.

        Args:
            hand_landmarks: (21, 4) array of hand landmarks
            pose_wrist: (3,) position of pose wrist

        Returns:
            Aligned hand landmarks with wrist at pose wrist position
        """
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return hand_landmarks

        aligned = hand_landmarks.copy()
        hand_wrist = aligned[HAND_WRIST, :3]
        offset = pose_wrist - hand_wrist
        aligned[:, :3] += offset
        return aligned

    def update(self, pose_result: LandmarkResult = None, hand_result: HandResult = None):
        """
        Update visualizer with new landmark results.

        Args:
            pose_result: LandmarkResult containing pose landmarks
            hand_result: HandResult containing left and right hand landmarks
        """
        # Get pose wrist positions for hand alignment
        pose_left_wrist = None
        pose_right_wrist = None
        pose_landmarks = None

        # Update pose
        if pose_result is not None and pose_result.world is not None:
            pose_landmarks = pose_result.world.values
            self._update_point_cloud(self.pose_pcd, pose_landmarks, [1, 0, 0])  # Red
            self._update_line_set(self.pose_lines, pose_landmarks, POSE_CONNECTIONS, [0, 1, 0])  # Green
            # Extract wrist positions for hand alignment
            if len(pose_landmarks) > POSE_LEFT_WRIST:
                pose_left_wrist = pose_landmarks[POSE_LEFT_WRIST, :3]
            if len(pose_landmarks) > POSE_RIGHT_WRIST:
                pose_right_wrist = pose_landmarks[POSE_RIGHT_WRIST, :3]
        else:
            self._update_point_cloud(self.pose_pcd, None, [1, 0, 0])
            self._update_line_set(self.pose_lines, None, POSE_CONNECTIONS, [0, 1, 0])

        # Update left hand
        if hand_result is not None and hand_result.left.world is not None:
            left_landmarks = hand_result.left.world.values
            # Align to pose wrist if available
            if pose_left_wrist is not None:
                left_landmarks = self._align_hand_to_pose_wrist(left_landmarks, pose_left_wrist)
            self._update_point_cloud(self.left_hand_pcd, left_landmarks, [0, 0, 1])  # Blue
            self._update_line_set(self.left_hand_lines, left_landmarks, HAND_CONNECTIONS, [0, 0.7, 1])  # Cyan
        else:
            self._update_point_cloud(self.left_hand_pcd, None, [0, 0, 1])
            self._update_line_set(self.left_hand_lines, None, HAND_CONNECTIONS, [0, 0.7, 1])

        # Update right hand
        if hand_result is not None and hand_result.right.world is not None:
            right_landmarks = hand_result.right.world.values
            # Align to pose wrist if available
            if pose_right_wrist is not None:
                right_landmarks = self._align_hand_to_pose_wrist(right_landmarks, pose_right_wrist)
            self._update_point_cloud(self.right_hand_pcd, right_landmarks, [1, 0.5, 0])  # Orange
            self._update_line_set(self.right_hand_lines, right_landmarks, HAND_CONNECTIONS, [1, 0.8, 0])  # Yellow
        else:
            self._update_point_cloud(self.right_hand_pcd, None, [1, 0.5, 0])
            self._update_line_set(self.right_hand_lines, None, HAND_CONNECTIONS, [1, 0.8, 0])

        # Update geometries in visualizer (only if points exist to avoid warnings)
        if len(self.pose_pcd.points) > 0:
            self.vis.update_geometry(self.pose_pcd)
        if len(self.pose_lines.points) > 0:
            self.vis.update_geometry(self.pose_lines)
        if len(self.left_hand_pcd.points) > 0:
            self.vis.update_geometry(self.left_hand_pcd)
        if len(self.left_hand_lines.points) > 0:
            self.vis.update_geometry(self.left_hand_lines)
        if len(self.right_hand_pcd.points) > 0:
            self.vis.update_geometry(self.right_hand_pcd)
        if len(self.right_hand_lines.points) > 0:
            self.vis.update_geometry(self.right_hand_lines)

        self.vis.poll_events()
        self.vis.update_renderer()

    def is_running(self) -> bool:
        """Check if the visualizer window is still open."""
        return self.vis.poll_events()

    def close(self):
        """Close the visualizer window."""
        self.vis.destroy_window()
