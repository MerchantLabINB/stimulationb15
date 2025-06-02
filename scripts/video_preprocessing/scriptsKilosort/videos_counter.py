import os

def count_videos(base_dir):
    """Counts the number of videos in Frontal and Lateral folders and their subdirectories."""
    frontal_count = 0
    lateral_count = 0

    # Walk through all directories and files
    for root, dirs, files in os.walk(base_dir):
        print(f"Checking directory: {root}")  # Debug output

        # Count video files in the current directory
        video_files = [file for file in files if file.endswith('.mp4')]
        if video_files:
            print(f"Found {len(video_files)} video(s) in {root}")  # Debug output

            # Update counts based on folder name
            if 'Frontal' in root:
                frontal_count += len(video_files)
                print(f"Added {len(video_files)} to Frontal count. Total: {frontal_count}")  # Debug output
            elif 'Lateral' in root:
                lateral_count += len(video_files)
                print(f"Added {len(video_files)} to Lateral count. Total: {lateral_count}")  # Debug output

    return frontal_count, lateral_count

if __name__ == "__main__":
    base_dir = r"C:\Users\Kilosort\Documents\Bruno\videosSegmentados"
    frontal_count, lateral_count = count_videos(base_dir)
    
    print(f"Number of videos in Frontal folders: {frontal_count}")
    print(f"Number of videos in Lateral folders: {lateral_count}")

    if frontal_count == lateral_count:
        print("The number of videos in Frontal and Lateral folders is the same.")
    else:
        print("The number of videos in Frontal and Lateral folders is different.")
