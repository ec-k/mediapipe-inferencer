import numpy as np
from pathlib import Path
from mediapipe_inferencer_core.filter.gaussian_1d_offline import Gaussian1dFilterOffline

if __name__ == '__main__':
    root_directory = str(Path(__file__).parent.parent)

    # NOTE: In my case, I want to use filtered results for ground-truth of SmoothNet
    raw_data = np.load(root_directory+'/data/saved/mediapipe_hand/detected/mediapipe_hand.npz')
    shape = raw_data['joints_3d'].shape
    ground_truth = {
        'joints_3d': np.ndarray(shape=shape, dtype='float32'),
        'imgname': raw_data['imgname']
    }
    
    joint = raw_data['joints_3d']

    sigma, window_size, n_landmarks = 3, 31, 21
    filter = Gaussian1dFilterOffline(sigma, window_size, n_landmarks)
    ground_truth['joints_3d'] = filter.filter(joint)
    np.savez(root_directory+'/data/saved/mediapipe_hand/groundtruth/mediapipe_hand_gaussian', joints_3d=ground_truth['joints_3d'], imgname=ground_truth['imgname'])
