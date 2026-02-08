from mediapipe_inferencer_core.network import HolisticPoseSender
from mediapipe_inferencer_core.detector import DetectorHandler, HandDetector, FaceDetector, PoseDetector
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core.image_provider import MmapImageProvider
from mediapipe_inferencer_core.filter import OneEuroFilter
from mmap_arg_parser import create_settings_from_args

from pathlib import Path
import cv2
import copy
import time
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

    pose_sender = HolisticPoseSender(config["pose_sender"]["host"], config["pose_sender"]["port"])
    pose_sender.connect()

    models_dir = str(base_dir / config["models_dir"])
    holistic_detector = DetectorHandler(
        pose=PoseDetector(models_dir + "/pose_landmarker_full.task", 0.8) if settings.enable_pose_inference else None,
        hand=HandDetector(models_dir + "/hand_landmarker.task", 0.8),
        face=FaceDetector(models_dir + "/face_landmarker.task", 0.8)
    )

    height, width = 720, 1280
    shape = (height, width, 4)

    filepath = Path(settings.mmap_file_path)
    if not filepath.exists():
        raise FileNotFoundError(f"The specified path does not exist: '{filepath}'")
    image_provider = MmapImageProvider(cache_queue_length=2, data_file_path=settings.mmap_file_path, shape=shape)

    min_cutoff, slope, d_min_cutoff = 1.0, 1.0, 1.0
    filter = {
        'left_hand_local':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'left_hand_world':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_local': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_world': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'face_landmark':    OneEuroFilter(min_cutoff, slope, d_min_cutoff)
        }
    if settings.enable_pose_inference:
        min_cutoff, slope = 0.08, 0.5
        filter['pose_local'] = OneEuroFilter(min_cutoff, slope, d_min_cutoff)
        filter['pose_world'] = OneEuroFilter(min_cutoff, slope, d_min_cutoff)

    while running:
        # Break in key Ctrl+C pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

        image = cv2.cvtColor(image_provider.latest_frame, cv2.COLOR_RGBA2BGR)
        if image is None:
            continue
        holistic_detector.inference(image)

        # Filtering (Note: use copy.deepcopy if you need to filter results)
        results = copy.deepcopy(holistic_detector.results)
        time_s = results.time
        results.hand.left.local = filter['left_hand_local'].filter(results.hand.left.local, time_s)
        results.hand.left.world = filter['left_hand_world'].filter(results.hand.left.world, time_s)
        results.hand.right.local = filter['right_hand_local'].filter(results.hand.right.local, time_s)
        results.hand.right.world = filter['right_hand_world'].filter(results.hand.right.world, time_s)
        results.face.landmarks = filter['face_landmark'].filter(results.face.landmarks, time_s)
        if settings.enable_pose_inference:
            results.pose.local = filter['pose_local'].filter(results.pose.local, time_s)
            results.pose.world = filter['pose_world'].filter(results.pose.world, time_s)

        # Send results to solver app
        pose_sender.send_holistic_landmarks(results)

        # Visualize resulted landmarks
        if settings.enable_visualization_window:
            annotated_image = image
            if results.pose is not None:
                annotated_image = visualizer.draw_pose_landmarks_on_image(annotated_image, results.pose)
            if results.hand is not None:
                annotated_image = visualizer.draw_hand_landmarks_on_image(annotated_image, results.hand)
            if results.face is not None:
                annotated_image = visualizer.draw_face_landmarks_on_image(annotated_image, results.face)
            # cv2.imshow('MediaPipe Landmarks', cv2.flip(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 1))
            cv2.imshow('MediaPipe Landmarks', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)) # no-flipping
        time.sleep(1/60)
    cv2.destroyAllWindows()