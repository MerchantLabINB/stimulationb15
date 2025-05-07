import os
import glob
import ffmpeg
from moviepy.editor import VideoFileClip
import pandas as pd
from datetime import datetime

def extract_video_info(directory, output_csv):
    # List all .mp4 files in the directory and its subdirectories
    video_files = glob.glob(os.path.join(directory, '**', '*.mp4'), recursive=True)

    # Initialize an empty list to store video information
    video_info = []

    # For each video file, extract information
    for video_file in video_files:
        try:
            clip = VideoFileClip(video_file)
            duration = clip.duration
            frame_rate = clip.fps
            size = clip.size
            file_size = os.path.getsize(video_file)
            creation_date = datetime.fromtimestamp(os.path.getctime(video_file)).strftime('%Y-%m-%d %H:%M:%S')
            modification_date = datetime.fromtimestamp(os.path.getmtime(video_file)).strftime('%Y-%m-%d %H:%M:%S')

            # Get video codec information using ffmpeg
            probe = ffmpeg.probe(video_file)
            video_codec = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')['codec_name']
            
            video_info.append([video_file, duration, frame_rate, size[0], size[1], file_size, creation_date, modification_date, video_codec, None])
        
        except Exception as e:
            print(f"Error processing file {video_file}: {e}")
            video_info.append([video_file, None, None, None, None, None, None, None, None, str(e)])

    # Create a DataFrame from the video information
    df = pd.DataFrame(video_info, columns=['File', 'Duration (s)', 'Frame Rate (fps)', 'Width (px)', 'Height (px)', 'File Size (bytes)', 'Creation Date', 'Modification Date', 'Video Codec', 'Error'])
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Video information saved to {output_csv}")
