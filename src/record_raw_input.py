# NOTE: I make this program for SmoothNet training

from mediapipe_inferencer_core.data_class import Landmark, LandmarkList
from mediapipe_inferencer_core.detector import DetectorHandler, HandDetector
from mediapipe_inferencer_core.filter.gaussian_1d_offline import Gaussian1dFilterOffline

import mediapipe as mp
from pathlib import Path
import cv2
import copy
import numpy as np


if __name__ == "__main__":
    root_directory = str(Path(__file__).parent.parent)

    video_file_path = root_directory + '/data/videos/video.mp4'
    cap_video = cv2.VideoCapture(video_file_path)
    n_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    n_landmarks = 21
    left_hand_result = {
        'joints_3d': np.ndarray(shape=(n_frames, n_landmarks * 3), dtype='float32'),
        'imgname': np.ndarray(shape=(n_frames), dtype=str),
    }
    right_hand_result = {
        'joints_3d': np.ndarray(shape=(n_frames, n_landmarks * 3), dtype='float32'),
        'imgname': np.ndarray(shape=(n_frames), dtype=str),
    }    

    holistic_detector = DetectorHandler(
        hand=HandDetector(
            root_directory + "/models/hand_landmarker.task",
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        ),
    )

    height, width = 720, 1280
    shape = (height, width, 4)
    project_root_directory = str(Path(__file__).parent.parent.parent)
    filepath = project_root_directory + "/colorImg.dat"
    # image_provider = MmapImageProvider(cache_queue_length=2, data_file_path=filepath, shape=shape)

    for i in range(n_frames):
        imgname = str(i)

        ret, image = cap_video.read()
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        # image = cv2.cvtColor(image_provider.latest_frame, cv2.COLOR_RGBA2BGR)
        if image is None:
            continue
        holistic_detector.inference(image)

        results = copy.deepcopy(holistic_detector.results)

        l_hand = None
        if results is not None and \
        results.hand is not None and \
        results.hand.left is not None and \
        results.hand.left.world is not None and\
        results.hand.left.world.values is not None:     
            l_hand = results.hand.left.world.values
        else:
            l_hand = LandmarkList([Landmark(0, 0, 0, 0) for _ in range(n_landmarks)]).values

        r_hand = None
        if results is not None and \
        results.hand is not None and \
        results.hand.right is not None and \
        results.hand.right.world is not None and\
        results.hand.right.world.values is not None:     
            r_hand = results.hand.right.world.values
        else:
            r_hand = LandmarkList([Landmark(0, 0, 0, 0) for _ in range(n_landmarks)]).values

        left_hand_result['imgname'][i] = right_hand_result['imgname'][i] = imgname
        # align landmarks with 1-dimention.
        left_hand_result['joints_3d'] = np.ravel(l_hand[:, :3])
        right_hand_result['joints_3d'] = np.ravel(r_hand[:, :3])
        
    cap_video.release()
    np.savez(root_directory+'/data/saved/mediapipe_hand/detected/mediapipe_left_hand', joints_3d=left_hand_result['joints_3d'], imgname=left_hand_result['imgname'])    
    np.savez(root_directory+'/data/saved/mediapipe_hand/detected/mediapipe_right_hand', joints_3d=right_hand_result['joints_3d'], imgname=right_hand_result['imgname'])    
