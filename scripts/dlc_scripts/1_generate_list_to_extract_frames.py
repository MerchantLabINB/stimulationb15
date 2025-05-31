import os
import random
from collections import defaultdict

# Directory containing all the videos
video_directory = r'C:\Users\Kilosort\Documents\Bruno\xv_lat-Br-2024-10-02\videos'

# Collect all video paths from the directory and subdirectories
videos_paths = []
for root, dirs, files in os.walk(video_directory):
    for file in files:
        if file.endswith(('.mp4', '.avi')):  # Adjust extensions as necessary
            videos_paths.append(os.path.join(root, file))

print(len(videos_paths))  # Print number of videos found

# Function to extract date from the filename (assuming '20240509' is in the format)
def extract_date(video_path):
    filename = os.path.basename(video_path)  # Get the file name
    return filename.split('__')[1][:8]  # Extract the '20240509' part of the file name

# Create a dictionary to group videos by their dates
videos_by_date = defaultdict(list)

for video in videos_paths:
    date = extract_date(video)
    videos_by_date[date].append(video)

# Randomly select videos ensuring diversity in dates
selected_videos = []
random.seed(42)  # Set seed for reproducibility

# Loop until we have selected 20 videos
while len(selected_videos) < 20:
    date = random.choice(list(videos_by_date.keys()))
    if videos_by_date[date]:  # Check if there are videos available for this date
        selected_video = random.choice(videos_by_date[date])
        selected_videos.append(selected_video)
        videos_by_date[date].remove(selected_video)  # Remove to avoid duplicates

# Print selected videos
print("Selected Videos:", selected_videos)

# Extract unique dates from the selected videos
unique_dates = set(extract_date(video) for video in selected_videos)

print("Unique Dates in Selected Videos:", unique_dates)
print("Number of Unique Dates:", len(unique_dates))

# Save the selected video paths to a text file in the desired format
with open('paths_for_frame_extraction.txt', 'w') as f:
    formatted_paths = ",\n".join([f'r"{video}"' for video in selected_videos])
    f.write(f"[{formatted_paths}]\n")  # Wrap in brackets for list format

print("Selected video paths saved to paths_for_frame_extraction.txt")
