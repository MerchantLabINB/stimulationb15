import os
import shutil

def move_lateral_videos(base_dir, target_dir):
    """Moves lateral videos from the base directory to the target directory."""
    lateral_count = 0

    # Walk through all directories and files
    for root, dirs, files in os.walk(base_dir):
        print(f"Checking directory: {root}")  # Debug output

        # Find and move lateral videos
        video_files = [file for file in files if file.endswith('.mp4')]
        if video_files and 'Lateral' in root:
            for video_file in video_files:
                source_path = os.path.join(root, video_file)
                destination_path = os.path.join(target_dir, video_file)
                shutil.move(source_path, destination_path)
                print(f"Moved {video_file} to {target_dir}")  # Debug output
                lateral_count += 1

    return lateral_count

if __name__ == "__main__":
    base_dir = r"C:\Users\Kilosort\Documents\Bruno\videosSegmentados"
    target_dir = r"C:\Users\Kilosort\Documents\Bruno\TesisXaviPoseEstimation(CamaraLateral)-BrunoBustos-2024-09-02\videos"
    
    lateral_count = move_lateral_videos(base_dir, target_dir)
    print(f"Total number of lateral videos moved: {lateral_count}")
