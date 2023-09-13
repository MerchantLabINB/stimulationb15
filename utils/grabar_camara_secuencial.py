from pypylon import pylon
import cv2
import time
import os
#
save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
# Getting the Transport Layer Factory
tlFactory = pylon.TlFactory.GetInstance()

# Get all attached devices
devices = tlFactory.EnumerateDevices()


for d in devices:
    print(d.GetModelName(), d.GetSerialNumber())


if len(devices) < 2:
    print("Not enough cameras present.")
    exit()

# Create and attach devices to camera objects
try:
    camera1 = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
    #width = camera1.Width.GetValue()
    #height = camera1.Height.GetValue()
    #print(f"Camera 1 Resolution: {width}x{height}")

except Exception as e:
    print("Error initializing Camera 1:", e)

try:
    camera2 = pylon.InstantCamera(tlFactory.CreateDevice(devices[1]))
    print("Camera 2 successfully initialized.")
except Exception as e:
    print("Error initializing Camera 2:", e)

# Initialize video writer using the mp4v codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# Define the subject, stimulation pattern, and get the current time for naming the output video files
subject_id = "subject_1"
stimulation_pattern = "pattern_A"
current_time = time.strftime("%Y%m%d-%H%M%S")

# Create VideoWriter objects for each camera to save the frames in MPEG4 format
video_writer1 = cv2.VideoWriter(os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam1.avi'), fourcc, 100.0, (1920, 1200))
video_writer2 = cv2.VideoWriter(os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam2.avi'), fourcc, 100.0, (1920, 1200))

print(video_writer1.isOpened())

# Set up the format converter for the grabbed images
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Start grabbing frames from both cameras using the 'LatestImageOnly' strategy
camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Loop to keep grabbing frames as long as both cameras are active
while camera1.IsGrabbing() and camera2.IsGrabbing():
    #print("Entrando al grabbing")
    # Retrieve the next image result from both cameras
    grabResult1 = camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    grabResult2 = camera2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    #print("SizeX: ", grabResult1.GetWidth())
    #print("SizeY: ", grabResult1.GetHeight())

    # Check if the frames were successfully grabbed
    if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():
        # Convert the grabbed images to OpenCV image format (BGR)
        img1 = converter.Convert(grabResult1).GetArray()
        img2 = converter.Convert(grabResult2).GetArray()

        # Write the images to the video files
        video_writer1.write(img1)
        video_writer2.write(img2)

        # Display the images in separate OpenCV windows
        cv2.imshow('Camera 1', img1)
        cv2.imshow('Camera 2', img2)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the grab results to free resources
    grabResult1.Release()
    grabResult2.Release()

# Release the video writer objects to save the videos and free resources
video_writer1.release()
video_writer2.release()

# Stop grabbing frames from the cameras
camera1.StopGrabbing()
camera2.StopGrabbing()

# Close all OpenCV windows
cv2.destroyAllWindows()