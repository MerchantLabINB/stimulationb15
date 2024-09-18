import subprocess
from utils.video_utils import get_bitrate  # Usamos la función de obtener el bitrate original

def reencode_video(input_path, output_path, bitrate=None, target_fps=30):
    """
    Re-encode a video to a target frame rate and optionally apply a specific bitrate.
    If no bitrate is provided, it will maintain the original bitrate.

    Parameters:
    - input_path: str, path to the input video file
    - output_path: str, path to the output video file
    - bitrate: str, optional, the target bitrate (e.g., '1000k' for 1000 kbps). If None, maintains original bitrate.
    - target_fps: int, target frames per second for the re-encoding process

    Returns:
    - None, but prints the status of the process.
    """
    try:
        # If no bitrate is provided, get the original bitrate of the video
        if not bitrate:
            original_bitrate = get_bitrate(input_path)
            if original_bitrate is None:
                print(f"Error: Could not retrieve the original bitrate for {input_path}.")
                return
            else:
                bitrate = original_bitrate
                print(f"Using original bitrate: {bitrate} for video {input_path}")

        # Build the ffmpeg command based on whether a bitrate is specified
        cmd = [
            'ffmpeg', '-i', input_path, '-r', str(target_fps), '-b:v', bitrate,
            '-c:a', 'copy', output_path
        ]

        # Run the ffmpeg command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if the command was successful
        if result.returncode == 0:
            print(f"Successfully re-encoded video {input_path} to {output_path} at {target_fps} fps with bitrate {bitrate}")
        else:
            print(f"Error re-encoding video {input_path}")
            print(result.stderr.decode('utf-8'))

    except Exception as e:
        print(f"An error occurred while re-encoding the video: {str(e)}")
