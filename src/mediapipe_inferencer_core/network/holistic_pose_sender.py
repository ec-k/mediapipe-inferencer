from mediapipe_inferencer_core.network.udp_client import UdpClient
from mediapipe_inferencer_core.packer.packer_for_sending import pack_holistic_landmarks_result
from mediapipe_inferencer_core.data_class.result_data import HolisticResults

class HolisticPoseSender:
    def __init__(self, address=None, port=None):
        self.client = UdpClient(address, port)

    def connect(self):
        self.client.connect()

    def send_holistic_landmarks(self, holistic_results:HolisticResults):
        """Send results with udp.

        Args:
            holistic_results (HolisticLandmarks): Packed results of MediaPipe Pose, Hands and Face.
        """
        packed_results = pack_holistic_landmarks_result(holistic_results)
        msg = packed_results.SerializeToString()
        return self.client.send_protobuf_message(msg)

