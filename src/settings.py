class Settings:
    def __init__(self,
                 mmap_file_path: str,
                 enable_visualization_window: bool,
                 enable_pose_inference: bool,
                 grpc_port: int = 50051,
                 preview_mmap_path: str = ""):
        self.mmap_file_path = mmap_file_path
        self.enable_visualization_window = enable_visualization_window
        self.enable_pose_inference = enable_pose_inference
        self.grpc_port = grpc_port
        self.preview_mmap_path = preview_mmap_path
