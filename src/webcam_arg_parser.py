import argparse
from dataclasses import dataclass


@dataclass
class WebcamSettings:
    enable_visualization_window: bool
    grpc_port: int
    preview_mmap_path: str
    preview_mmap_alpha: bool


def create_settings_from_args() -> WebcamSettings:
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable_visualization_window', action='store_true')
    parser.add_argument('--grpc_port', type=int, default=50051)
    parser.add_argument('--preview_mmap_path', default="")
    parser.add_argument('--preview_mmap_alpha', action='store_true')

    args = parser.parse_args()

    return WebcamSettings(
        enable_visualization_window=args.enable_visualization_window,
        grpc_port=args.grpc_port,
        preview_mmap_path=args.preview_mmap_path,
        preview_mmap_alpha=args.preview_mmap_alpha
    )
