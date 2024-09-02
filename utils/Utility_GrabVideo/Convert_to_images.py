import cv2

# Open the video file
video = cv2.VideoCapture('output.mp4')

# Initialize a counter for the frame number
frame_number = 0

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # If the frame was not successfully read, then we have reached the end of the video
    if not ret:
        break

    # Save the frame as an image file
    cv2.imwrite(f'frame{frame_number}.png', frame)

    # Increment the frame number
    frame_number += 1

# Release the video file
print ("Total frames: ", frame_number)
video.release()