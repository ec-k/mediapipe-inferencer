from mediapipe_inferencer_core.network.udp_client import UdpClient
from mediapipe_inferencer_core.packer.packer_for_sending import pack_holistic_landmarks_result
from mediapipe_inferencer_core.data_class import HolisticResults, LandmarkList, Landmark
import copy

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
        result = copy.deepcopy(holistic_results)
        if holistic_results.pose is not None and holistic_results.pose.world is not None:
            result.pose.world = transform_coordinate(holistic_results.pose.world)
        packed_results = pack_holistic_landmarks_result(result)
        msg = packed_results.SerializeToString()
        return self.client.send_protobuf_message(msg)


def transform_coordinate(result: LandmarkList):
    return LandmarkList(Landmark(-lm[0], -lm[1], -lm[2], lm[3]) for lm in result.values)

