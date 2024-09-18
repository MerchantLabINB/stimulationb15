# utils/ubuntu/video_utils.py

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