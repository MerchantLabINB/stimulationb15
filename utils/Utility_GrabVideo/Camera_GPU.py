from pypylon import pylon
import subprocess
import time

def save_video_with_gpu(camera, duration_seconds, video_path):
    
    frame_rate = 30  # Adjust based on camera capabilities and desired output

    ffmpeg_command = [
    'ffmpeg',
    '-y',  # Overwrite output files without asking
    '-f', 'rawvideo',  # Input format
    '-vcodec', 'rawvideo',
    '-s', '1920x1200',  # Updated size of one frame
    '-pix_fmt', 'bgr24',  # Assuming BGR format for the input
    '-r', str(frame_rate),  # Frames per second
    '-i', '-',  # The input comes from a pipe
    '-c:v', 'h264_nvenc',  # Use GPU-accelerated video encoding
    '-pix_fmt', 'yuv420p',  # Output pixel format
    video_path
]

    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

    start_time = time.time()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    while camera.IsGrabbing() and time.time() - start_time < duration_seconds:
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            # Convert grabbed frame to a numpy array
            img = grab_result.GetArray()
            # Print the shape of the image array
            print("Resolution: ", img.shape)

            # Print the data type of the image array
            print("Pixel format: ", img.dtype)
            # Ensure the frame is written in the correct byte size matching the expected resolution and pixel format
            process.stdin.write(img.tobytes())
        grab_result.Release()

    camera.StopGrabbing()
    process.stdin.close()
    process.wait()

# Initialize and configure camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()


# Refer to your camera's specific SDK
# Make sure to set the camera to the desired resolution, frame rate, etc., here
# This might include camera.PixelFormat.SetValue("BGR8"), camera.Width.SetValue(1920), camera.Height.SetValue(1200), etc.
# Refer to your camera's specific SDK documentation for the exact method names and parameters

video_path = 'output_gpu.mp4'
duration_seconds = 5

save_video_with_gpu(camera, duration_seconds, video_path)

camera.Close()