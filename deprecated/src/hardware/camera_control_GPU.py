import cv2
from pypylon import pylon
import threading

def capture_and_process(camera_index, output_path, stop_event):
    """
    Captures video from a camera and saves the video. Demonstrates basic structure without GPU processing.
    
    Args:
    - camera_index: Index of the camera to use.
    - output_path: Path to save the output video file.
    - stop_event: Threading event to stop the loop.
    """
    # Initialize the camera
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if len(devices) == 0:
        raise RuntimeError("No camera detected.")
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[camera_index]))
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # Configure converter for OpenCV compatibility
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Define the codec and create VideoWriter object
    # Note: Ensure the frame size matches your camera's output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 120.0, (1280, 720))

    while not stop_event.is_set():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grab_result)
            frame = image.GetArray()

            # If you need to process the frame, do it here
            # For GPU processing, you would transfer frame to GPU, process it, and then transfer it back
            # frame_processed = some_gpu_processing_function(frame)

            # Write the frame (or processed frame) to the video file
            out.write(frame)

        grab_result.Release()

    camera.StopGrabbing()
    camera.Close()
    out.release()

if __name__ == "__main__":
    stop_event = threading.Event()
    try:
        threads = [
            threading.Thread(target=capture_and_process, args=(0, 'output1.avi', stop_event)),
            threading.Thread(target=capture_and_process, args=(1, 'output2.avi', stop_event))
        ]

        for thread in threads:
            thread.start()

        input("Press Enter to stop recording...\n")

    finally:
        stop_event.set()
        for thread in threads:
            thread.join()

        print("Finished recording.")
