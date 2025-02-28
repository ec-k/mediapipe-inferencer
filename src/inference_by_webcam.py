from mediapipe_inferencer_core.network.holistic_pose_sender import HolisticPoseSender
from mediapipe_inferencer_core.detector.holistic_detector import HolisticDetector
from mediapipe_inferencer_core import visualizer
from mediapipe_inferencer_core import packer

import cv2
import mediapipe as mp
import numpy as np
import time


if __name__ == "__main__":
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_holistic = mp.solutions.holistic

  pose_sender = HolisticPoseSender()
  pose_sender.connect()

  holistic_detector = HolisticDetector()

  cap = cv2.VideoCapture(0)
  while cap.isOpened():
      # Break in key Ctrl+C pressed
      if cv2.waitKey(5) & 0xFF == 27:
          break

      # Inference pose
      success, image = cap.read()
      if not success:
          print("Capturing is failed.")
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

