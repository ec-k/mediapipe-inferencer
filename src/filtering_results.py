import numpy as np
from pathlib import Path
from mediapipe_inferencer_core.filter.gaussian_1d_offline import Gaussian1dFilterOffline

if __name__ == '__main__':
    root_directory = str(Path(__file__).parent.parent)

    # NOTE: In my case, I want to use filtered results for ground-truth of SmoothNet
    raw_data_left = np.load(root_directory+'/data/saved/mediapipe_hand/detected/mediapipe_left_hand.npz')
    raw_data_right = np.load(root_directory+'/data/saved/mediapipe_hand/detected/mediapipe_right_hand.npz')
    shape = raw_data_left['joints_3d'].shape
    ground_truth_left = {
        'joints_3d': np.ndarray(shape=shape, dtype='float32'),
        'joints_2d': np.ndarray(shape=shape, dtype='float32'),
        'imgname': raw_data_left['imgname']
    }
    ground_truth_right={
        'joints_3d': np.ndarray(shape=shape, dtype='float32'),
        'joints_2d': np.ndarray(shape=shape, dtype='float32'),
        'imgname': raw_data_left['imgname']
    }

    sigma, window_size, n_landmarks = 3, 31, 21
    filter = Gaussian1dFilterOffline(sigma, window_size, n_landmarks)
    ground_truth_left['joints_3d'] = filter.filter(raw_data_left['joints_3d'])
    ground_truth_left['joints_2d'] = filter.filter(raw_data_left['joints_2d'])
    ground_truth_right['joints_3d'] = filter.filter(raw_data_right['joints_3d'])
    ground_truth_right['joints_2d'] = filter.filter(raw_data_right['joints_2d'])
    np.savez(root_directory+'/data/saved/mediapipe_hand/groundtruth/mediapipe_left_hand_gaussian',
             joints_3d=ground_truth_left['joints_3d'],
             joints_2d=ground_truth_left['joints_2d'],
             imgname=ground_truth_left['imgname'])
    np.savez(root_directory+'/data/saved/mediapipe_hand/groundtruth/mediapipe_right_hand_gaussian',
             joints_3d=ground_truth_right['joints_3d'],
             joints_2d=ground_truth_right['joints_2d'],
             imgname=ground_truth_right['imgname'])
