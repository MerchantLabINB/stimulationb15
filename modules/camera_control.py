from pypylon import pylon
import cv2
import time
import os

# Definiendo variables importantes

camera1 = None
camera2 = None
video_writer1 = None
video_writer2 = None


def init_cameras():
    # Initialization logic for camera1 and camera2

    # Getting the Transport Layer Factory
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices
    try:
        devices = tlFactory.EnumerateDevices()
    except Exception as e:
        print(f"Failed to enumerate devices: {e}")
        exit()

    # Check if cameras are present
    if len(devices) < 2:
        print("Not enough cameras present.")
        exit()

    # Create and attach devices to camera objects
    try:
        camera1 = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
    except pylon.RuntimeException as e:
        print(f"Runtime error initializing Camera 1: {e}")
    except pylon.GenericException as e:
        print(f"Generic error initializing Camera 1: {e}")

    try:
        camera2 = pylon.InstantCamera(tlFactory.CreateDevice(devices[1]))
    except pylon.RuntimeException as e:
        print(f"Runtime error initializing Camera 2: {e}")
    except pylon.GenericException as e:
        print(f"Generic error initializing Camera 2: {e}")

    return camera1, camera2



def grab_and_write_frames(camera, converter, video_writer):
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        img = converter.Convert(grabResult).GetArray()
        video_writer.write(img)
        cv2.imshow(f'Camera {camera}', img)
    grabResult.Release()

def start_record(camera1, camera2, save_dir, subject_id, stimulation_pattern,frame_rate = 120):
    # Initialize video writer using the mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')


    # Define the subject, stimulation pattern, and get the current time for naming the output video files

    current_time = time.strftime("%Y%m%d-%H%M%S")

    # Create VideoWriter objects for each camera to save the frames in MPEG4 format
    video_writer1 = cv2.VideoWriter(os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam1.avi'), fourcc, frame_rate, (1920, 1200))
    video_writer2 = cv2.VideoWriter(os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam2.avi'), fourcc, frame_rate, (1920, 1200))

    # print(video_writer1.isOpened())

    # Set up the format converter for the grabbed images
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Start grabbing frames from both cameras using the 'LatestImageOnly' strategy
    camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Loop to keep grabbing frames as long as both cameras are active
    while camera1.IsGrabbing() and camera2.IsGrabbing():
        grab_and_write_frames(camera1, converter, video_writer1)
        grab_and_write_frames(camera2, converter, video_writer2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return video_writer1, video_writer2

def stop_record(video_writer1, video_writer2, camera1, camera2):
    # Release the video writer objects to save the videos and free resources
    video_writer1.release()
    video_writer2.release()

    # Stop grabbing frames from the cameras
    camera1.StopGrabbing()
    camera2.StopGrabbing()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'

    camera1, camera2 = init_cameras()
    subject_id = 'prueba_script'
    stimulation_pattern = ''
    video_writer1, video_writer2 = start_record(camera1, camera2, save_dir, subject_id, stimulation_pattern)
    stop_record(video_writer1, video_writer2, camera1, camera2)