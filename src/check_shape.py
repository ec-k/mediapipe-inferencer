import numpy as np

left_data = np.load('./data/saved/mediapipe_hand/detected/mediapipe_left_hand.npz')
right_data = np.load('./data/saved/mediapipe_hand/detected/mediapipe_right_hand.npz')

print('raw_data')
print('left')
print(f'files: {left_data.files}')
print(f'shape: {left_data["joints_3d"].shape}')

print('right')
print(f'files: {right_data.files}')
print(f'shape: {right_data["joints_3d"].shape}')

print()
print()

left_data = np.load('./data/saved/mediapipe_hand/groundtruth/mediapipe_left_hand_gaussian.npz')
right_data = np.load('./data/saved/mediapipe_hand/groundtruth/mediapipe_right_hand_gaussian.npz')

print('filtered_data')
print('left')
print(f'files: {left_data.files}')
print(f'shape: {left_data["joints_3d"].shape}')

print('right')
print(f'files: {right_data.files}')
print(f'shape: {right_data["joints_3d"].shape}')
