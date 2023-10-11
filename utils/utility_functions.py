from pypylon import pylon
import cv2
import time
import os
import numpy as np
from collections import deque
import threading

class CameraControl:
    def __init__(self):
        self.camera1 = None
        self.camera2 = None
        self.video_writer1 = None
        self.video_writer2 = None
        self.converter1 = None
        self.converter2 = None
        self.frame_buffer1 = deque(maxlen=100)  # Adjust the buffer size as needed
        self.frame_buffer2 = deque(maxlen=100)
        self.display_thread = None

        # Create locks for camera access
        self.camera_lock1 = threading.Lock()
        self.camera_lock2 = threading.Lock()

        self.buffer_lock = threading.Lock()

    def init_cameras(self):
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            print("Cámaras inicializadas")

            if len(devices) < 2:
                print("Not enough cameras present.")
                return False

            self.camera1 = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
            self.camera2 = pylon.InstantCamera(tlFactory.CreateDevice(devices[1]))
            print("Cámaras inicializadas")
            return True
        except Exception as e:
            print(f"Failed to initialize cameras: {e}")
            return False

    def display_frames(self):
        while True:
            if not self.camera1 or not self.camera2:
                print("Cameras are not initialized.")
                continue

            if not self.camera1.IsGrabbing() or not self.camera2.IsGrabbing():
                print("Cameras are not grabbing frames.")
                continue

            try:
                grabResult1 = self.camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                grabResult2 = self.camera2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():
                    img1 = self.converter1.Convert(grabResult1).GetArray()
                    img2 = self.converter2.Convert(grabResult2).GetArray()

                    # Display the resized frames for live viewing
                    cv2.imshow('Cameras 1 & 2', np.hstack((img1, img2)))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Pressed 'q' within display_frames loop")
                        break

                grabResult1.Release()
                grabResult2.Release()
            except Exception as e:
                print(f"Error while displaying frames: {e}")

    def start_display_thread(self):
        # Create and start the display thread
        self.display_thread = threading.Thread(target=self.display_frames)
        self.display_thread.daemon = True
        self.display_thread.start()

    def stop_display_thread(self):
        if self.display_thread:
            self.display_thread.join()

    def grab_frames(self):
        while True:
            if not self.camera1 or not self.camera2:
                print("Cameras are not initialized.")
                continue

            if not self.camera1.IsGrabbing() or not self.camera2.IsGrabbing():
                print("Cameras are not grabbing frames.")
                continue

            try:
                grabResult1 = self.camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                grabResult2 = self.camera2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():
                    img1 = self.converter1.Convert(grabResult1).GetArray()
                    img2 = self.converter2.Convert(grabResult2).GetArray()

                    # Use locks to ensure thread safety while appending frames to the buffer
                    with self.camera_lock1:
                        with self.buffer_lock:
                            self.frame_buffer1.append(img1)
                    with self.camera_lock2:
                        with self.buffer_lock:
                            self.frame_buffer2.append(img2)

                grabResult1.Release()
                grabResult2.Release()
            except Exception as e:
                print(f"Error while grabbing frames: {e}")

    def buffer_frames(self):
        while True:
            with self.buffer_lock:
                if len(self.frame_buffer1) == self.frame_buffer1.maxlen:
                    # Write frames from the buffer to the video file
                    for i in range(len(self.frame_buffer1)):
                        self.video_writer1.write(self.frame_buffer1[i])
                        self.video_writer2.write(self.frame_buffer2[i])

                    # Clear the buffer
                    self.frame_buffer1.clear()
                    self.frame_buffer2.clear()

    def start_record_with_buffering(self, save_dir, subject_id, stimulation_pattern, recording_duration, frame_rate=100):
        if not self.camera1 or not self.camera2:
            print("Cameras are not initialized.")
            return

        try:
            # Initialize video writers using the mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG' or other codecs

            current_time = time.strftime("%Y%m%d-%H%M%S")

            file_name1 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam1.avi')
            file_name2 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam2.avi')

            # Calculate the number of frames required based on the desired recording duration
            num_frames = int(frame_rate * recording_duration)
            print("Se grabará a", frame_rate, "frames por segundo")
            print("Se grabarán un total de", num_frames, "frames")

            self.video_writer1 = cv2.VideoWriter(file_name1, fourcc, frame_rate, (1920, 1200))
            self.video_writer2 = cv2.VideoWriter(file_name2, fourcc, frame_rate, (1920, 1200))

            self.converter1 = pylon.ImageFormatConverter()
            self.converter1.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter1.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.converter2 = pylon.ImageFormatConverter()
            self.converter2.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter2.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            print("Video writers and grabbers are initialized")

            frame_count = 0  # Counter for the captured frames
            start_time = time.time()  # Record the start time

            # Start the frame grabbing thread
            frame_grabbing_thread = threading.Thread(target=self.grab_frames)
            frame_grabbing_thread.daemon = True
            frame_grabbing_thread.start()

            # Start the frame buffering thread
            frame_buffering_thread = threading.Thread(target=self.buffer_frames)
            frame_buffering_thread.daemon = True
            frame_buffering_thread.start()

            while (self.camera1.IsGrabbing() and self.camera2.IsGrabbing() and
                    frame_count < num_frames):
                # Display the resized frames for live viewing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Pressed 'q' within start_record loop")
                    break

                # Check if the recording duration has elapsed
                elapsed_time = time.time() - start_time
                if elapsed_time >= recording_duration:
                    break

            # Write any remaining frames in the buffer
            for i in range(len(self.frame_buffer1)):
                self.video_writer1.write(self.frame_buffer1[i])
                self.video_writer2.write(self.frame_buffer2[i])

            # Clear the buffer
            self.frame_buffer1.clear()
            self.frame_buffer2.clear()

            print("Frame count after the loop: ", frame_count)
        except Exception as e:
            print(f"Error while starting recording: {e}")

    def close_cv2_windows(self):
        cv2.destroyAllWindows()

    def stop_record(self):
        try:
            if self.video_writer1 and self.video_writer2:
                self.video_writer1.release()
                self.video_writer2.release()

            if self.camera1 and self.camera2:
                self.camera1.StopGrabbing()
                self.camera2.StopGrabbing()
        except Exception as e:
            print(f"Error while stopping recording: {e}")

    def check_frame_rate_for_cameras(self):
        try:
            if not self.camera1 or not self.camera2:
                print("Cameras are not initialized.")
                return

            # Open the cameras for configuration
            self.camera1.Open()
            self.camera2.Open()

            # Access the camera's NodeMap for settings
            node_map1 = self.camera1.GetNodeMap()
            node_map2 = self.camera2.GetNodeMap()

            # Query the Acquisition Frame Rate
            acquisition_frame_rate1 = node_map1.GetNode("AcquisitionFrameRate")
            acquisition_frame_rate2 = node_map2.GetNode("AcquisitionFrameRate")

            # Get the current frame rates
            current_frame_rate1 = acquisition_frame_rate1.GetValue()
            current_frame_rate2 = acquisition_frame_rate2.GetValue()

            print(f"Camera 1 Frame Rate: {current_frame_rate1} FPS")
            print(f"Camera 2 Frame Rate: {current_frame_rate2} FPS")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Close the cameras
            self.camera1.Close()
            self.camera2.Close()

if __name__ == "__main__":
    save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
    subject_id = 'prueba_script'
    stimulation_pattern = ''

    # Create an instance of CameraControl
    camera_controller = CameraControl()

    # Initialize the cameras
    if camera_controller.init_cameras():
        try:
            # Start the display thread
            camera_controller.start_display_thread()

            for _ in range(3):  # Repeat the functioning 3 times
                # Start recording for each trial with different parameters
                recording_duration = 3  # seconds (trial duration)

                camera_controller.start_record_with_buffering(save_dir, subject_id, stimulation_pattern,
                                                             recording_duration, frame_rate=100)  # Use the modified method

                # Close the OpenCV window before stopping recording
                camera_controller.close_cv2_windows()

                # Stop recording for the current trial
                camera_controller.stop_record()

                # Add intertrial time here (randomly generated)
                intertrial_time = 2
                time.sleep(intertrial_time)

            print("Program executed successfully\n")
        except KeyboardInterrupt:
            # Exit the loop when the user presses Ctrl+C
            pass
        finally:
            # Stop the display thread
            camera_controller.stop_display_thread()

"""
def grab_and_write_frames(self, show_video=True):
        print("THIS SHOULDNT BE CALLED")
        if not self.camera1 or not self.camera2:
            print("Cameras are not initialized.")
            return
        
        try:
            # Start grabbing frames from both cameras 
            if not self.camera1.IsGrabbing() and not self.camera2.IsGrabbing():
                self.camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


            grabResult1 = self.camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            grabResult2 = self.camera2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():
                img1 = self.converter1.Convert(grabResult1).GetArray()
                img2 = self.converter2.Convert(grabResult2).GetArray()
                
                # Resize the frames for live display (adjust dimensions as needed)
                img1_resized = cv2.resize(img1, (640, 480))
                img2_resized = cv2.resize(img2, (640, 480))
                
                # Display the resized frames for live viewing
                if show_video:
                    cv2.imshow('Cameras 1 & 2', np.hstack((img1_resized, img2_resized)))

                # Save the original frames to videos
                self.video_writer1.write(img1)
                self.video_writer2.write(img2)

            grabResult1.Release()
            grabResult2.Release()
        except Exception as e:
            print(f"Error while grabbing and writing frames: {e}")

    def start_record(self, save_dir, subject_id, stimulation_pattern, recording_duration, frame_rate=100):
        if not self.camera1 or not self.camera2:
            print("Cameras are not initialized.")
            return

        try:
            # Initialize video writers using the mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG' or other codecs

            current_time = time.strftime("%Y%m%d-%H%M%S")

            file_name1 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam1.avi')
            file_name2 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam2.avi')

            # Calculate the number of frames required based on the desired recording duration
            num_frames = int(frame_rate * recording_duration)
            print("Se grabará a",frame_rate,"frames por segundo")
            print("Se grabarán un total de",num_frames,"frames")

            self.video_writer1 = cv2.VideoWriter(file_name1, fourcc, frame_rate, (1920, 1200))
            self.video_writer2 = cv2.VideoWriter(file_name2, fourcc, frame_rate, (1920, 1200))

            self.converter1 = pylon.ImageFormatConverter()
            self.converter1.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter1.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.converter2 = pylon.ImageFormatConverter()
            self.converter2.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter2.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            print("Video writers and grabbers are initialized")

            frame_count = 0  # Counter for the captured frames
            start_time = time.time()  # Record the start time

            while (self.camera1.IsGrabbing() and self.camera2.IsGrabbing() and
                    frame_count < num_frames):
                self.grab_and_write_frames(show_video=True)
                frame_count += 1  # Increment the frame count

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Pressed 'q' within start_record loop")
                    break

                # Check if the recording duration has elapsed
                elapsed_time = time.time() - start_time
                if elapsed_time >= recording_duration:
                    break

            print("Frame count after the loop: ", frame_count)
        except Exception as e:
            print(f"Error while starting recording: {e}")

if __name__ == "__main__":
    save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
    subject_id = 'prueba_script'
    stimulation_pattern = ''

    # Create an instance of CameraControl
    camera_controller = CameraControl()

    # Initialize the cameras
    if camera_controller.init_cameras():
        try:
            # camera_controller.check_frame_rate_for_cameras()
            for _ in range(3):  # Repeat the functioning 7 times
                # Start recording for each trial with different parameters
                recording_duration = 3  # seconds (trial duration)

                camera_controller.start_record(save_dir, subject_id, stimulation_pattern,recording_duration,frame_rate=100)                
                #  Close the OpenCV window before stopping recording
                camera_controller.close_cv2_windows()

                # Stop recording for the current trial
                camera_controller.stop_record()

                # Add intertrial time here (randomly generated)
                #intertrial_time = random.uniform(1.0, 5.0)  # Adjust the range as needed
                intertrial_time = 2
                time.sleep(intertrial_time)
            
            print("Programa ejecutado correctamente\n")
        except KeyboardInterrupt:
            # Exit the loop when the user presses Ctrl+C
            pass
"""