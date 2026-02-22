import grpc
from concurrent import futures

from proto_generated import inferencer_control_pb2_grpc as pb2_grpc
from .estimation_state import EstimationState
from .estimation_control_servicer import EstimationControlServicer


class EstimationControlServer:
    """gRPC server wrapper for EstimationControl service."""

    def __init__(self, state: EstimationState, port: int, max_workers: int = 2):
        self._state = state
        self._port = port
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        self._servicer = EstimationControlServicer(state)
        pb2_grpc.add_EstimationControlServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(f'[::]:{port}')

    def start(self):
        """Start the gRPC server."""
        self._server.start()

    def stop(self, grace: float = 0.5):
        """Stop the gRPC server."""
        self._server.stop(grace)

    def wait_for_termination(self, timeout: float = None):
        """Block until the server stops."""
        self._server.wait_for_termination(timeout)

    @property
    def port(self) -> int:
        return self._port
