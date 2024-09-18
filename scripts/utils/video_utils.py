import subprocess
import json

def get_frame_rate(video_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'json', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output = json.loads(result.stdout)
        if 'streams' in output and len(output['streams']) > 0:
            r_frame_rate = output['streams'][0]['r_frame_rate']
            num, denom = map(int, r_frame_rate.split('/'))
            frame_rate = num / denom
            return frame_rate
        else:
            return None
    except json.JSONDecodeError:
        return None
    
def get_bitrate(video_path):
    """
    Retrieves the bitrate of the video using ffprobe.
    
    Parameters:
    - video_path: str, path to the video file
    
    Returns:
    - str: bitrate of the video (e.g., '1000k'), or None if an error occurred
    """
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=bit_rate', '-of', 'json', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        output = json.loads(result.stdout)
        if 'format' in output and 'bit_rate' in output['format']:
            # Bitrate is in bits per second, so convert it to 'k' format (e.g., '1000k')
            bitrate = int(output['format']['bit_rate']) // 1000  # Convert to kbps
            return f'{bitrate}k'
        else:
            print(f"Error: Could not find 'bit_rate' in the output for {video_path}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from ffprobe output: {e}")
        return None

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
            bitrate = get_bitrate(input_path)
            if bitrate is None:
                print(f"Error: Could not retrieve the original bitrate for {input_path}.")
                return
            else:
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
