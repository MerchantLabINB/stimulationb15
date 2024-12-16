import argparse
import os
from utils.video_utils import get_bitrate, get_frame_rate, reencode_video

def main():
    """
    You can run the script as follows:


    python reencode_videos.py --input-dir /path/to/input/videos --output-dir /path/to/output/videos --fps 100

    """
    parser = argparse.ArgumentParser(description="Preprocess videos by re-encoding.")
    parser.add_argument('--input-dir', required=True, help='Path to the input directory with videos')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory for reencoded videos')
    parser.add_argument('--fps', type=int, default=100, help='Target frame rate for re-encoding')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    target_fps = args.fps

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the input directory to find MP4 files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, f'reencoded_{file}')
                
                print(f"Processing video: {video_path}")

                # Get original bitrate
                bitrate = get_bitrate(video_path)
                if bitrate is None:
                    print(f"Skipping video {video_path}: could not retrieve original bitrate.")
                    continue  # Skip this file if bitrate retrieval fails

                # Get original frame rate
                actual_fps = get_frame_rate(video_path)
                if actual_fps is None:
                    print(f"Skipping video {video_path}: could not retrieve frame rate.")
                    continue  # Skip this file if frame rate retrieval fails

                # Re-encode video
                try:
                    reencode_video(video_path, output_path, bitrate, target_fps)
                except Exception as e:
                    print(f"Error re-encoding video {video_path}: {e}")
                    continue  # Skip this file if re-encoding fails

if __name__ == "__main__":
    main()
