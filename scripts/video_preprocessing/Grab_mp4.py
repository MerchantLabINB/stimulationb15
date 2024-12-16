from pypylon import pylon
import cv2

def grab_video_from_camera(output_file, video_duration=5, frame_rate=120):
    # Calculate the total number of frames to capture
    total_frames = video_duration * frame_rate
    
    # Connect to the camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Set camera parameters
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(frame_rate)

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (camera.Width.GetValue(), camera.Height.GetValue()))

    # Start grabbing frames
    captured_frames = 0
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    while camera.IsGrabbing() and captured_frames < total_frames:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        captured_frames += 1
        
        if grabResult.GrabSucceeded():
            frame = grabResult.Array
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
    
    camera.StopGrabbing()
    out.release()
    camera.Close()
    print(f"Video saved: {output_file}")
