from mediapipe_inferencer_core.util.float_util import clamp, lerp

def test_clamp():
    assert clamp(-1, 0, 1) == 0
    assert clamp(0.3, 0, 1) == 0.3
    assert clamp(1.4, 0, 1) == 1

def test_float_lerp():
    assert lerp(0, 1, -1) == 0
    assert lerp(0, 1, 0) == 0
    assert lerp(0, 1, 0.4) == 0.4
    assert lerp(0, 1, 1) == 1
    assert lerp(0, 1, 2) == 1
