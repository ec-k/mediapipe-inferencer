from mediapipe_inferencer_core.data_class.landmark import Landmark
from mediapipe_inferencer_core.util import float_util

def test_value_equal():
    landmark_1 = Landmark(0, 1, 2, 3)
    landmark_2 = Landmark(1, 1, 2, 2)
    value_equal = Landmark.value_equal
    assert value_equal(landmark_1, landmark_1) == True
    assert value_equal(landmark_1, landmark_2) == False

def test_ladmark_lerp():
    lerp = Landmark.lerp
    value_equal = Landmark.value_equal
    landmark_1 = Landmark(0, 1, 2, 0)
    landmark_2 = Landmark(1, 2, 3, 1)
    assert value_equal(lerp(landmark_1, landmark_2, -1), landmark_1)
    assert value_equal(lerp(landmark_1, landmark_2, 0), landmark_1)
    alpha = 0.3
    x = float_util.lerp(landmark_1.x, landmark_2.x, alpha)
    y = float_util.lerp(landmark_1.y, landmark_2.y, alpha)
    z = float_util.lerp(landmark_1.z, landmark_2.z, alpha)
    confidence = float_util.lerp(landmark_1.confidence, landmark_2.confidence, alpha)
    assert value_equal(lerp(landmark_1, landmark_2, 0.3), Landmark(x, y, z, confidence))
    assert value_equal(lerp(landmark_1, landmark_2, 1), landmark_2)
    assert value_equal(lerp(landmark_1, landmark_2, 2), landmark_2)
