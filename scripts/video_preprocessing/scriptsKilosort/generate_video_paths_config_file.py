import os

# Define the path to the directory containing your videos
video_dir = r'C:\Users\Kilosort\Documents\Bruno\TesisXaviPoseEstimation(CamaraLateral)-BrunoBustos-2024-09-02\videos'

# Define the output file path
output_file = 'video_paths.txt'

# List all files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

# Write the video paths to the file
with open(output_file, 'w') as f:
    for i, video_file in enumerate(video_files, start=1):
        full_path = os.path.join(video_dir, video_file)
        f.write(f'video{i}: \n  - {full_path}\n')

print(f'Video paths have been written to {output_file}')
