import argparse
from dataclasses import dataclass


@dataclass
class MmapSettings:
    mmap_file_path: str
    enable_visualization_window: bool
    enable_pose_inference: bool


def create_settings_from_args() -> MmapSettings:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mmap_file_path', default="")
    parser.add_argument('--enable_pose_inference', action='store_true')
    parser.add_argument('--enable_visualization_window', action='store_true')

    args = parser.parse_args()

    return MmapSettings(
        mmap_file_path=args.mmap_file_path,
        enable_visualization_window=args.enable_visualization_window,
        enable_pose_inference=args.enable_pose_inference
    )
