import cv2
import os

def convert_video_to_images(video_path, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Initialize a counter for the frame number
    frame_number = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If the frame was not successfully read, then we have reached the end of the video
        if not ret:
            break

        # Save the frame as an image file
        frame_path = os.path.join(output_dir, f'frame{frame_number}.png')
        cv2.imwrite(frame_path, frame)

        # Increment the frame number
        frame_number += 1

    # Release the video file
    video.release()
    print(f"Total frames: {frame_number}")
