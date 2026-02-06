from mediapipe_inferencer_core.network import HolisticPoseSender, EstimationState, EstimationControlServer
from mediapipe_inferencer_core.detector import DetectorHandler, PoseDetector, HandDetector, FaceDetector
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core.image_provider import WebcamImageProvider, MmapImageWriter
from mediapipe_inferencer_core.filter import OneEuroFilter
from app_arg_parser import create_settings_from_args

import cv2
import time
import copy
from pathlib import Path
import signal
import sys
import json


def get_base_directory() -> Path:
    try:
        # Nuitka compiled
        return Path(__nuitka_binary_dir)
    except NameError:
        # Development environment
        return Path(__file__).parent.parent


def load_config(base_dir: Path) -> dict:
    settings_path = base_dir / "settings.json"
    if not settings_path.exists():
        print(f"Error: settings.json not found at {settings_path}", file=sys.stderr)
        sys.exit(1)
    with open(settings_path) as f:
        return json.load(f)


def create_filters(enable_pose: bool):
    min_cutoff, slope, d_min_cutoff = 1.0, 4.0, 1.0
    filters = {
        'left_hand_local':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'left_hand_world':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_local': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_world': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'face_landmark':    OneEuroFilter(min_cutoff, slope, d_min_cutoff)
    }
    if enable_pose:
        min_cutoff, slope = 0.08, 0.5
        filters['pose_local'] = OneEuroFilter(min_cutoff, slope, d_min_cutoff)
        filters['pose_world'] = OneEuroFilter(min_cutoff, slope, d_min_cutoff)
    return filters


running = True

def handle_sigint(signum, frame):
    global running
    running = False


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    settings = create_settings_from_args()

    base_dir = get_base_directory()
    config = load_config(base_dir)

    # Initialize estimation state and gRPC server
    estimation_state = EstimationState()
    estimation_state.set_landmark_visualization(
        pose=settings.enable_pose_inference,
        hands=True,
        face=True
    )

    grpc_server = EstimationControlServer(estimation_state, settings.grpc_port)
    grpc_server.start()
    print(f"gRPC server started on port {settings.grpc_port}")

    pose_sender = HolisticPoseSender(config["pose_sender"]["host"], config["pose_sender"]["port"])
    pose_sender.connect()

    models_dir = str(base_dir / config["models_dir"])
    holistic_detector = DetectorHandler(
        pose=PoseDetector(models_dir + "/pose_landmarker_full.task", 0.8) if settings.enable_pose_inference else None,
        hand=HandDetector(models_dir + "/hand_landmarker.task", 0.8),
        face=FaceDetector(models_dir + "/face_landmarker.task", 0.8)
    )

    image_provider = WebcamImageProvider(cache_queue_length=2, device_index=estimation_state.get_camera_index())
    filters = create_filters(settings.enable_pose_inference)

    # Initialize preview writer if path is specified
    preview_writer = None
    if settings.preview_mmap_path:
        preview_shape = (720, 1280, 4)  # BGRA format
        preview_writer = MmapImageWriter(settings.preview_mmap_path, preview_shape)
        print(f"Preview mmap writer initialized: {settings.preview_mmap_path}")

    # Auto-start estimation
    estimation_state.set_running(True)

    while running:
        # Check for stop request
        if estimation_state.stop_requested.is_set():
            estimation_state.set_running(False)

        # Check for start request
        if estimation_state.start_requested.is_set():
            estimation_state.set_running(True)

        # Handle camera change
        if estimation_state.camera_change_requested.is_set():
            image_provider.release_capture()
            image_provider = WebcamImageProvider(
                cache_queue_length=2,
                device_index=estimation_state.get_camera_index()
            )
            estimation_state.acknowledge_camera_change()

        # Break on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Skip inference if not running
        if not estimation_state.is_running:
            time.sleep(1/60)
            continue

        # Inference pose
        image_provider.update()
        image = cv2.cvtColor(image_provider.latest_frame, cv2.COLOR_RGBA2BGR)
        if image is None:
            continue
        holistic_detector.inference(image)

        if holistic_detector.results is None:
            continue

        # Filtering
        results = copy.deepcopy(holistic_detector.results)
        time_s = results.time
        results.hand.left.local = filters['left_hand_local'].filter(results.hand.left.local, time_s)
        results.hand.left.world = filters['left_hand_world'].filter(results.hand.left.world, time_s)
        results.hand.right.local = filters['right_hand_local'].filter(results.hand.right.local, time_s)
        results.hand.right.world = filters['right_hand_world'].filter(results.hand.right.world, time_s)
        results.face.landmarks = filters['face_landmark'].filter(results.face.landmarks, time_s)
        if settings.enable_pose_inference:
            results.pose.local = filters['pose_local'].filter(results.pose.local, time_s)
            results.pose.world = filters['pose_world'].filter(results.pose.world, time_s)

        # Send results to solver app
        pose_sender.send_holistic_landmarks(results)

        # Visualize resulted landmarks
        viz_settings = estimation_state.get_landmark_visualization()
        annotated_image = image
        if results.pose is not None and viz_settings.pose_enabled:
            annotated_image = visualizer.draw_pose_landmarks_on_image(annotated_image, results.pose)
        if results.hand is not None and viz_settings.hands_enabled:
            annotated_image = visualizer.draw_hand_landmarks_on_image(annotated_image, results.hand)
        if results.face is not None and viz_settings.face_enabled:
            annotated_image = visualizer.draw_face_landmarks_on_image(annotated_image, results.face)

        if settings.enable_visualization_window:
            cv2.imshow('MediaPipe Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        # Write to shared memory if preview is enabled
        if preview_writer and estimation_state.get_preview_enabled():
            preview_writer.write(annotated_image)

        time.sleep(1/60)

    # Cleanup
    grpc_server.stop()
    image_provider.release_capture()
    if preview_writer:
        preview_writer.close()
    cv2.destroyAllWindows()
