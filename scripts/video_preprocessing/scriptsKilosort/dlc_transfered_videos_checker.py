import os
import pandas as pd

# Load the CSV file
csv_file_path = 'extracted_segments_info.csv'  # Update this path if necessary
df = pd.read_csv(csv_file_path)

# Extract the list of lateral videos from the CSV (strip whitespace and handle case sensitivity)
lateral_videos_from_csv = df["Lateral Video"].apply(lambda x: os.path.basename(x).strip().lower()).tolist()
print(lateral_videos_from_csv)
# Path to the folder where the videos are supposed to be transferred
transferred_videos_dir = r"C:\Users\Kilosort\Documents\Bruno\TesisXaviPoseEstimation(CamaraLateral)-BrunoBustos-2024-09-02\videos"

# Get the list of .mp4 files in the transferred videos directory (also lowercase for case-insensitivity)
def get_transferred_mp4_files(directory):
    mp4_files = [f.lower().strip() for f in os.listdir(directory) if f.endswith('.mp4')]
    
    # Print each file for debugging
    """
    print("\nFiles found in the directory:")
    for file in mp4_files:
        print(file)
    """
    
    return mp4_files

# Get transferred videos
transferred_videos = get_transferred_mp4_files(transferred_videos_dir)

# Find the common files and missing files
transferred_set = set(transferred_videos)  # Set of videos found in the directory
csv_set = set(lateral_videos_from_csv)     # Set of videos listed in the CSV

"""
# Debugging: Print the actual video files being compared
print("\nVideos in CSV:")
for video in lateral_videos_from_csv:
    print(video)
"""
# Find coincidences (files present in both the CSV and the directory)
coincidences = csv_set.intersection(transferred_set)

# Find videos that are missing from the directory but listed in the CSV
not_transferred = csv_set.difference(transferred_set)

# Debugging: Print each filename being compared
print("\nFiles found in the directory:")
for file in transferred_videos:
    print(file)

print("\nVideos listed in CSV:")
for video in lateral_videos_from_csv:
    print(video)

# Output the results
print("\nTransferred Videos (Coincidences):")
for video in sorted(coincidences):
    print(video)

print("\nNot Transferred Videos (Missing):")
for video in sorted(not_transferred):
    print(video)

# Output the counts
print(f"\nTotal videos in CSV: {len(lateral_videos_from_csv)}")
print(f"Total transferred videos found in the directory: {len(transferred_videos)}")
print(f"Total matching videos: {len(coincidences)}")
print(f"Total missing videos: {len(not_transferred)}")
