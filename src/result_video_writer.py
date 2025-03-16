# Read .npz and write annotated video

import numpy as np
from pathlib import Path
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core.data_class import LandmarkList, Landmark, HandResult
import cv2

def pack_to_LandmarkList(array_data: np.ndarray)->LandmarkList:
    # Suppose that array_data is flattened
    n = len(array_data) // 3
    return LandmarkList([Landmark(array_data[3*i], array_data[3*i+1], array_data[3*i+2], 1) for i in range(n)])

if __name__ == '__main__':
    root_directory = str(Path(__file__).parent.parent)

    video_file_path = root_directory + '/data/videos/typing_writing_paper_note.mp4'
    cap_video = cv2.VideoCapture(video_file_path)
    video_properties={
        'width': int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': float(cap_video.get(cv2.CAP_PROP_FPS))
    }
    n_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    n_landmarks = 21

    landmark_data_path = {
        'left': root_directory + '/data/saved/mediapipe_hand/groundtruth/mediapipe_left_hand_gaussian.npz',
        'right': root_directory + '/data/saved/mediapipe_hand/groundtruth/mediapipe_right_hand_gaussian.npz',
    }

    data = {
        'left':np.load(landmark_data_path['left']),
        'right': np.load(landmark_data_path['right']),
    }
    results = {
        'left':data['left']['joints_2d'],
        'right':data['right']['joints_2d'],
    }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = root_directory + '/data/videos/annotated_video.mp4'
    # out_path = 'annotated_video.mp4'
    video_writer = cv2.VideoWriter(out_path, fourcc, video_properties['fps'], (video_properties['width'], video_properties['height']), True)
    for i in range(n_frames):
        # Inference pose
        ret, image = cap_video.read()
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if image is None:
            continue

        one_frame_result = {
            'left': pack_to_LandmarkList(results['left'][i]),
            'right': pack_to_LandmarkList(results['right'][i]),
        }

        # Visualize resulted landmarks
        annotated_image = image
        if results is not None:
            annotated_image = visualizer.draw_hand_landmarks_on_image_from_LandmarkList(annotated_image, one_frame_result['left'], one_frame_result['right'])
        video_writer.write(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    cap_video.release()
    video_writer.release()