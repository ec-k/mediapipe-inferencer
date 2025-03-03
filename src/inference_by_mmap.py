from mediapipe_inferencer_core.network.holistic_pose_sender import HolisticPoseSender
from mediapipe_inferencer_core.detector.detector_handler import DetectorHandler
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core.image_provider import MmapImageProvider
from mediapipe_inferencer_core.detector.landmark_detector import HandDetector, FaceDetector
from mediapipe_inferencer_core.filter.one_euro_filter import OneEuroFilter

from pathlib import Path
import cv2
import copy
import time


if __name__ == "__main__":
    pose_sender = HolisticPoseSender("localhost", 9001)
    pose_sender.connect()

    root_directory = str(Path(__file__).parent.parent)
    holistic_detector = DetectorHandler(
        hand=HandDetector(root_directory + "/models/hand_landmarker.task"),
        face=FaceDetector(root_directory + "/models/face_landmarker.task")
    )

    height, width = 720, 1280
    shape = (height, width, 4)
    project_root_directory = str(Path(__file__).parent.parent.parent)
    filepath = project_root_directory + "/colorImg.dat"
    image_provider = MmapImageProvider(cache_queue_length=2, data_file_path=filepath, shape=shape)

    min_cutoff, slope, d_min_cutoff = 1.0, 4, 1.0
    filter = {
        'left_hand_local':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'left_hand_world':  OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_local': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'right_hand_world': OneEuroFilter(min_cutoff, slope, d_min_cutoff),
        'face_landmark':    OneEuroFilter(min_cutoff, slope, d_min_cutoff)
        }
    while True:
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

        # Send results to solver app
        pose_sender.send_holistic_landmarks(results)

        # Visualize resulted landmarks
        annotated_image = image
        if results.hand is not None:
            annotated_image = visualizer.draw_hand_landmarks_on_image(annotated_image, results.hand)
        if results.face is not None:
            annotated_image = visualizer.draw_face_landmarks_on_image(annotated_image, results.face)
        cv2.imshow('MediaPipe Landmarks', cv2.flip(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 1))
        time.sleep(1/60)
    cv2.destroyAllWindows()