from mediapipe_inferencer_core.network.udp_client import UdpClient
from proto.holistic_landmarks_pb2 import HolisticLandmarks

class HolisticPoseSender:
    def __init__(self, address=None, port=None):
        self.client = UdpClient(address, port)

    def connect(self):
        self.client.connect()

    def send_holistic_landmarks(self, holistic_results:HolisticLandmarks):
        """Send results with udp.

        Args:
            holistic_results (HolisticLandmarks): Packed results of MediaPipe Pose, Hands and Face.
        """
        msg = holistic_results.SerializeToString()
        return self.client.send_protobuf_message(msg)

