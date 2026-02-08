import argparse
from dataclasses import dataclass


@dataclass
class WebcamSettings:
    enable_visualization_window: bool
    enable_pose_inference: bool
    grpc_port: int
    preview_mmap_path: str


def create_settings_from_args() -> WebcamSettings:
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable_pose_inference', action='store_true')
    parser.add_argument('--enable_visualization_window', action='store_true')
    parser.add_argument('--grpc_port', type=int, default=50051)
    parser.add_argument('--preview_mmap_path', default="")

    args = parser.parse_args()

    return WebcamSettings(
        enable_visualization_window=args.enable_visualization_window,
        enable_pose_inference=args.enable_pose_inference,
        grpc_port=args.grpc_port,
        preview_mmap_path=args.preview_mmap_path
    )
