import os
import re

def extract_camera_id_and_date(folder_name):
    """Extracts the full camera ID and date with microseconds from the folder name."""
    match = re.search(r'(\d{8}_\d{9})', folder_name)  # Capture full timestamp with 9 digits
    camera_id_match = re.search(r'__([\d]+)__', folder_name)
    
    if match and camera_id_match:
        full_date = match.group(1)
        camera_id = camera_id_match.group(1)
        return camera_id, full_date
    return None, None

def rename_videos_in_directory(base_dir):
    """Renames videos in the specified directory by replacing the partial camera ID and date with the full one."""
    for root, dirs, files in os.walk(base_dir):
        # Skip directories that don't contain videos
        if not any(file.endswith('.mp4') for file in files):
            continue

        # Extract camera ID and full date from the folder name (root)
        folder_name = os.path.basename(root)
        camera_id, full_date = extract_camera_id_and_date(folder_name)
        if not full_date or not camera_id:
            print(f"No valid camera ID or full date pattern found in folder name: {folder_name}")
            continue

        # Rename each video file
        for file_name in files:
            if file_name.endswith('.mp4'):
                old_file_path = os.path.join(root, file_name)

                # Replace the partial date (8 digits date + 6 digits time) with the full date (9 digits)
                new_file_name = re.sub(r'(\d{8}_\d{6})', f'{camera_id}__{full_date}', file_name)

                new_file_path = os.path.join(root, new_file_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")

if __name__ == "__main__":
    base_dir = r"C:\Users\Kilosort\Documents\Bruno\videosSegmentados"
    rename_videos_in_directory(base_dir)
