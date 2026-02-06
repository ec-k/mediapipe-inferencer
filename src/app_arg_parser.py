from settings import Settings
import argparse

def create_settings_from_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mmap_file_path', default="")
    parser.add_argument('--enable_pose_inference', action='store_true')
    parser.add_argument('--enable_visualization_window', action='store_true')
    parser.add_argument('--grpc_port', type=int, default=50051)
    parser.add_argument('--preview_mmap_path', default="")

    args = parser.parse_args()

    settings = Settings(
        args.mmap_file_path,
        args.enable_visualization_window,
        args.enable_pose_inference,
        args.grpc_port,
        args.preview_mmap_path
    )

    return settings
