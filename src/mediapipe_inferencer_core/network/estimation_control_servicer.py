from proto_generated import inferencer_control_pb2 as pb2
from proto_generated import inferencer_control_pb2_grpc as pb2_grpc
from .estimation_state import EstimationState


class EstimationControlServicer(pb2_grpc.EstimationControlServicer):
    """gRPC service implementation for estimation control."""

    def __init__(self, state: EstimationState):
        self._state = state

    def SelectCamera(self, request, context):
        self._state.set_camera_index(request.device_index)
        return pb2.OperationResult(success=True)

    def SetPreviewEnabled(self, request, context):
        self._state.set_preview_enabled(request.enabled)
        return pb2.OperationResult(success=True)

    def SetLandmarkVisualization(self, request, context):
        self._state.set_landmark_visualization(
            pose=request.pose_enabled,
            hands=request.hands_enabled,
            face=request.face_enabled
        )
        return pb2.OperationResult(success=True)

    def GetStatus(self, request, context):
        viz = self._state.get_landmark_visualization()
        return pb2.EstimationStatus(
            is_running=self._state.is_running,
            selected_camera_index=self._state.get_camera_index(),
            preview_enabled=self._state.get_preview_enabled(),
            landmark_visualization=pb2.LandmarkVisualizationSettings(
                pose_enabled=viz.pose_enabled,
                hands_enabled=viz.hands_enabled,
                face_enabled=viz.face_enabled
            )
        )

    def Start(self, request, context):
        if self._state.request_start():
            return pb2.OperationResult(success=True)
        return pb2.OperationResult(success=False, error_message="Already running")

    def Stop(self, request, context):
        if self._state.request_stop():
            return pb2.OperationResult(success=True)
        return pb2.OperationResult(success=False, error_message="Not running")
