import os
import cv2

# Define the input and output directories
input_dir = 'videosPreprocesados'
output_dir = 'videosPequenhos'

# Iterate over all the files in the input directory
for root, dirs, files in os.walk(input_dir):
    for file in files:
        # Get the full path of the input video
        input_path = os.path.join(root, file)
        
        # Create the corresponding output directory structure
        output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
        os.makedirs(output_subdir, exist_ok=True)
        
        # Get the full path of the output video
        output_path = os.path.join(output_subdir, file)
        
        # Open the input video
        video = cv2.VideoCapture(input_path)
        
        # Check if the video opened successfully
        if not video.isOpened():
            print(f"Error: Could not open video file {input_path}")
            continue
        
        # Get the original video dimensions
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Check if width and height are valid
        if width == 0 or height == 0:
            print(f"Error: Invalid dimensions for video file {input_path}")
            video.release()
            continue
        
        # Get the original frames per second (fps)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # Calculate the new dimensions while maintaining the aspect ratio
        if width > height:
            new_width = 640
            new_height = int(height * (640 / width))
        else:
            new_width = int(width * (640 / height))
            new_height = 640
        
        # Create a VideoWriter object to write the resized video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        # Read and resize each frame of the input video
        while True:
            ret, frame = video.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (new_width, new_height))
            output_video.write(resized_frame)
        
        # Release the video objects
        video.release()
        output_video.release()

print("Videos resized and saved successfully!")
