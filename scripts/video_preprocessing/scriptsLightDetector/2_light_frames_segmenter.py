import os
import cv2
import pandas as pd
import numpy as np

# Define the ROI for each date
roi_dict = {
    '09/05': (147, 1334, 86, 80),
    '15/05': (131, 1200, 86, 80),
    '18/05': (134, 1049, 86, 80),
    '23/05': (134, 953, 86, 80),
    '24/05': (144, 953, 86, 80),
    '28/05': (227, 1155, 86, 80)
}

# Load the CSV file
csv_path = "Stimuli_information.csv"
print(f"Loading CSV file from {csv_path}")
data = pd.read_csv(csv_path)
print("First few rows of the dataset:")
print(data.head())

# Extract relevant part of the filename, handle multiple lines by splitting and selecting the correct one
def extract_filename_prefix(x):
    if isinstance(x, str):
        identifiers = x.splitlines()
        for identifier in identifiers:
            if '40298451' in identifier:
                return identifier.split('__')[1].strip()  # Extract date and time part only
    return None

data['Filename_Prefix'] = data['Archivos de video'].apply(extract_filename_prefix)

print(f"Unique dates in 'Día experimental': {data['Día experimental'].unique()}")

# Filter data only for dates that have a defined ROI
filtered_data = data[
    (data['Filename_Prefix'].notna()) &
    (data['Día experimental'].isin(roi_dict.keys()))
]

print(f"Found {len(filtered_data)} records matching the criteria.")

# Set the base directory for videos
video_dir = os.getcwd()
print(f"Video directory set to: {video_dir}")

# Create directories for saving extracted frames
os.makedirs('light_frames', exist_ok=True)
os.makedirs('no_light_frames', exist_ok=True)
print("Directories 'light_frames' and 'no_light_frames' created.")

# CSV to log the frame extraction details
log_csv_path = "extracted_frames_log.csv"
log_columns = ["Filename", "Type", "Video Identifier", "Frame Index"]
frame_log = []

# Function to find the video path
def find_video_path(video_dir, identifier):
    print(f"Searching in directory: {video_dir} for files containing: {identifier}")
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if identifier in file:
                full_path = os.path.join(root, file)
                print(f"Video file found: {full_path}")
                return full_path
    print(f"No video file found with identifier: {identifier}")
    return None

# Function to extract and save frames using the ROI
def extract_and_save_frames(video_path, start_frame, end_frame, label, count, video_identifier, roi):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return count
    
    frame_idx = 0
    x, y, w, h = roi  # Unpack the ROI

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frames in the region of interest
        if start_frame <= frame_idx <= end_frame:
            frame_roi = frame[y:y+h, x:x+w]  # Crop the frame to the ROI
            frame_name = f"{label}_{video_identifier}_frame_{frame_idx}.png"
            if label == 'light':
                save_path = os.path.join('light_frames', frame_name)
            else:
                save_path = os.path.join('no_light_frames', frame_name)
            cv2.imwrite(save_path, frame_roi)
            print(f"Frame {label} saved: {save_path}")
            frame_log.append([frame_name, label, video_identifier, frame_idx])
            count += 1
        
        frame_idx += 1
    
    cap.release()
    return count

light_frame_count = 0

# Loop through each video file and extract frames
for index, row in filtered_data.iterrows():
    video_file = row['Archivos de video']
    start_frame_str = row['Start frame (lateral)']
    duration = row['Duración (ms)']
    dia_experimental = row['Día experimental']
    
    # Use the ROI corresponding to the date of the experiment
    roi = roi_dict[dia_experimental]
    
    if pd.isna(start_frame_str) or pd.isna(duration):
        print(f"Skipping row {index} due to missing start frame or duration.")
        continue
    
    video_identifier = f"40298451__{row['Filename_Prefix']}"
    print(f"Constructed video identifier: {video_identifier}")
    
    video_path = find_video_path(video_dir, video_identifier)
    print(f"Video path found: {video_path}")
    
    if not video_path:
        print(f"Video file with identifier '{video_identifier}' not found.")
        continue
    
    print(f"Processing file: {video_path}")
    
    try:
        start_frame_list = [int(x) for x in start_frame_str.split(',') if x.strip().isdigit()]
        duration_frames = int(duration / 10)  # Convert duration from ms to frames (100 fps)
    except ValueError as e:
        print(f"Error processing frames for record {index}: {e}")
        continue
    
    print(f"Start frames: {start_frame_list}, Duration (in frames): {duration_frames}")
    
    used_frames = set()

    for start_frame in start_frame_list:
        end_frame = start_frame + duration_frames - 1
        print(f"Extracting light frames from {start_frame} to {end_frame} in '{video_file}'")
        
        light_frame_count = extract_and_save_frames(video_path, start_frame, end_frame, 'light', light_frame_count, video_identifier, roi)
        
        used_frames.update(range(start_frame, end_frame + 1))

    # Select no-light frames excluding the light frames
    all_frames = set(range(int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))))
    available_no_light_frames = sorted(list(all_frames - used_frames))

    if len(available_no_light_frames) >= duration_frames:
        no_light_start = np.random.choice(available_no_light_frames[:-duration_frames+1])
        no_light_end = no_light_start + duration_frames - 1
        print(f"Extracting no-light frames from {no_light_start} to {no_light_end} in '{video_file}'")
        extract_and_save_frames(video_path, no_light_start, no_light_end, 'no_light', 0, video_identifier, roi)

# Save the frame extraction log to a CSV
log_df = pd.DataFrame(frame_log, columns=log_columns)
log_df.to_csv(log_csv_path, index=False)
print(f"Extraction log saved to {log_csv_path}")

print(f"Extracted {light_frame_count} light frames and saved an equal number of no-light frames.")
