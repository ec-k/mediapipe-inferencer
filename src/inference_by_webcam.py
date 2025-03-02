from mediapipe_inferencer_core.network.holistic_pose_sender import HolisticPoseSender
from mediapipe_inferencer_core.detector.detector_handler import DetectorHandler
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core import packer
from mediapipe_inferencer_core.iamge_provider import WebcamImageProvider
from mediapipe_inferencer_core.detector.landmark_detector import PoseDetector, HandDetector, FaceDetector

import cv2
import time
from pathlib import Path


if __name__ == "__main__":
  pose_sender = HolisticPoseSender()
  pose_sender.connect()

  root_directory = str(Path(__file__).parent.parent)
  holistic_detector = DetectorHandler(
    pose=PoseDetector(root_directory + "/models/pose_landmarker_full.task"),
    hand=HandDetector(root_directory + "/models/hand_landmarker.task"),
    face=FaceDetector(root_directory + "/models/face_landmarker.task")
  )

  cap = cv2.VideoCapture(0)
  image_provider = WebcamImageProvider(cache_queue_length=2, device_index=0)
  while image_provider.is_opened:
      # Break in key Ctrl+C pressed
      if cv2.waitKey(5) & 0xFF == 27:
          break

      # Inference pose
      image_provider.update()
      image = image_provider.latest_frame
      if image is None:
          continue
      holistic_detector.inference(image)

        # Filtering (Note: use copy.deepcopy if you need to filter results)
      results = holistic_detector.results

      # Send results to solver app
      packed_results = packer.pack_holistic_landmarks_result(results)
      pose_sender.send_holistic_landmarks(packed_results)

      # Visualize resulted landmarks
      annotated_image = image
      if holistic_detector.results.pose_landmarks is not None:
        annotated_image = visualizer.draw_pose_landmarks_on_image(annotated_image, results.pose_landmarks)
      if results.hand_landmarks is not None:
        annotated_image = visualizer.draw_hand_landmarks_on_image(annotated_image, results.hand_landmarks)
      if results.face_results is not None:
        annotated_image = visualizer.draw_face_landmarks_on_image(annotated_image, results.face_results)
      cv2.imshow('MediaPipe Landmarks', cv2.flip(annotated_image, 1))
      time.sleep(1/60)
  cap.release()

