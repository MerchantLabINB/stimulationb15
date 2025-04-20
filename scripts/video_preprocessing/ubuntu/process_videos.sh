#!/bin/bash

# Define the input and output directories
input_folder="/mnt/c/Users/samae/Documents/GitHub/GUI_pattern_generator/data/datos_tesis/videosXavy/mayo24/Lateral"
output_folder="/mnt/c/Users/samae/Documents/GitHub/GUI_pattern_generator/data/datos_tesis/normalizacion100fps"
log_file="$output_folder/processing_log.txt"

# Ensure the output folder exists
mkdir -p "$output_folder"

# Clear the log file
> "$log_file"

# Desired fps
desired_fps=100

# Process each .mp4 file in the input folder
for input_file in "$input_folder"/*.mp4; do
    filename=$(basename "$input_file")
    output_file="$output_folder/${filename%.*}_${desired_fps}fps.mp4"
    
    # Get the original fps and duration
    original_fps=$(ffmpeg -i "$input_file" 2>&1 | grep -oP '(?<=, )\d+(?= fps)')
    original_duration=$(ffmpeg -i "$input_file" 2>&1 | grep -oP '(?<=Duration: )[^,]*')
    
    if [ -z "$original_fps" ]; then
        echo "Could not determine the original fps for $input_file" | tee -a "$log_file"
        continue
    fi

    # Calculate the original duration in seconds
    IFS=: read -r h m s <<< "$original_duration"
    total_seconds=$(echo "$h*3600 + $m*60 + $s" | bc)
    
    # Calculate the number of frames
    num_frames=$(echo "$original_fps * $total_seconds" | bc)
    
    # Calculate the new duration
    new_duration=$(echo "scale=2; $num_frames / $desired_fps" | bc)
    
    # Calculate the factor by which to adjust the timestamps
    adjust_factor=$(echo "scale=4; $original_fps / $desired_fps" | bc)

    # Log the important values
    echo "Processing $input_file" | tee -a "$log_file"
    echo "Original FPS: $original_fps" | tee -a "$log_file"
    echo "Desired FPS: $desired_fps" | tee -a "$log_file"
    echo "Original Duration: $original_duration ($total_seconds seconds)" | tee -a "$log_file"
    echo "Number of Frames: $num_frames" | tee -a "$log_file"
    echo "New Duration: $new_duration seconds" | tee -a "$log_file"
    echo "Adjust Factor: $adjust_factor" | tee -a "$log_file"
    echo "Output File: $output_file" | tee -a "$log_file"
    
    # Use ffmpeg to change the FPS and adjust the duration to keep all frames
    if ffmpeg -i "$input_file" -filter:v "setpts=PTS*$adjust_factor" -r $desired_fps "$output_file"; then
        echo "Processed $input_file to $output_file with $desired_fps fps" | tee -a "$log_file"
    else
        echo "Failed to process $input_file" | tee -a "$log_file"
    fi
done
