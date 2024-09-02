from pypylon import pylon
import cv2

# Set the desired video duration and frame rate
video_duration = 5  # duration in seconds
frame_rate = 120  # frames per second

# Calculate the total number of frames to capture
total_frames = video_duration * frame_rate

# Connect to the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set camera parameters
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(frame_rate)  # Set the desired frame rate

# Create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (camera.Width.GetValue(), camera.Height.GetValue()))

# Initialize a counter for the number of captured frames
captured_frames = 0

# Start grabbing frames from the camera
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

print("Capturing video...")
while camera.IsGrabbing() and captured_frames < total_frames:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    # Increment the counter
    captured_frames += 1
    
    if grabResult.GrabSucceeded():
        # Convert the grabbed frame to OpenCV format
        frame = grabResult.Array
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write the frame to the video file
        out.write(frame)

    grabResult.Release()

# Release the camera and close the video file
camera.StopGrabbing()
camera.Close()
out.release()
