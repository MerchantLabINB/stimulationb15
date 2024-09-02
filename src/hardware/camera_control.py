# camera_control.py
import sys
import os

# Append the project's root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import threading
import cv2
import time
import os
import random
import datetime
import numpy as np
from pypylon import pylon
from utils.utility_functions import run_c_program

lock = threading.Lock()

class CameraControl:
    def __init__(self):
        self.camera1 = None
        self.camera2 = None
        self.video_writer1 = None
        self.video_writer2 = None
        self.converter1 = None
        self.converter2 = None
        self.stop_event = threading.Event()  # Initialize the stop event

    def send_ttl_signal(self, value, duration):
        try:
            # Acquire the thread lock before sending TTL signal
            with lock:
                # Send TTL signal here with both value and duration
                output = run_c_program(value, duration)
                if output is not None:
                    print("TTL Signal Sent:", output)
                else:
                    print("Error sending TTL signal.")
        except Exception as e:
            print(f"Error sending TTL signal: {e}")
    def send_ttl_signal_threaded(self, value, duration):
        threading.Thread(target=self.send_ttl_signal, args=(value, duration)).start()


    def init_cameras(self):
        if self.camera1 is not None or self.camera2 is not None:
            print("Cámaras ya inicializadas.")
            return True
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if len(devices) < 2:
                print("Not enough cameras present.")
                return False

            self.camera1 = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
            self.camera2 = pylon.InstantCamera(tlFactory.CreateDevice(devices[1]))
            
            # Open the cameras for configuration
            self.camera1.Open()
            self.camera2.Open()

            print("Cámaras inicializadas")
            return True
        
        except Exception as e:
            print(f"Failed to initialize cameras: {e}")
            return False

    """
    def grab_and_write_frames(self, show_video=True):
        if not self.camera1 or not self.camera2:
            print("Cameras are not initialized.")
            return

        frame_count = 0  # Initialize frame count here

        try:
            if not self.camera1.IsGrabbing() and not self.camera2.IsGrabbing():
                self.camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            while (self.camera1.IsGrabbing() and self.camera2.IsGrabbing()):
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
                        cv2.waitKey(1)  # Update the OpenCV window

                    # Save the original frames to videos
                    self.video_writer1.write(img1)
                    self.video_writer2.write(img2)

                    frame_count += 1  # Increment frame count here

                grabResult1.Release()
                grabResult2.Release()
        except Exception as e:
            print(f"Error while grabbing and writing frames: {e}")

        return frame_count  # Return the updated frame count
    """
    def start_record(self, save_dir, subject_id, stimulation_pattern, recording_duration, frame_rate=120):
        if not self.camera1 or not self.camera2:
            print("Cameras are not initialized.")
            return

        try:
            # Initialize video writers using the XVID codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            current_time = time.strftime("%Y%m%d-%H%M%S")

            file_name1 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam1.avi')
            file_name2 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam2.avi')

            # Create VideoWriter objects with only the codec and frame rate
            self.video_writer1 = cv2.VideoWriter(file_name1, fourcc, frame_rate, (480,640))
            self.video_writer2 = cv2.VideoWriter(file_name2, fourcc, frame_rate, (480,640))

            self.converter1 = pylon.ImageFormatConverter()
            self.converter1.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter1.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            self.converter2 = pylon.ImageFormatConverter()
            self.converter2.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter2.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            # Start the grabbing process
            self.camera1.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.camera2.StartGrabbing(pylon.GrabStrategy_OneByOne)

            print("Video writers and grabbers are initialized")

            frame_count = 0  # Counter for the captured frames
            start_time = time.time()  # Record the start time
            ttl_signal_sent = False
            expected_frame_count = int(frame_rate * recording_duration)

            while frame_count < expected_frame_count:
                grabResult1 = self.camera1.RetrieveResult(50, pylon.TimeoutHandling_ThrowException)
                grabResult2 = self.camera2.RetrieveResult(50, pylon.TimeoutHandling_ThrowException)

                if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():
                    img1 = self.converter1.Convert(grabResult1).GetArray()
                    img2 = self.converter2.Convert(grabResult2).GetArray()

                    # Resize the frames for live display (adjust dimensions as needed)
                    img1_resized = cv2.resize(img1, (640,480))
                    img2_resized = cv2.resize(img2, (640,480))

                    # Rotate the frames as needed
                    img1_rotated = cv2.rotate(img1_resized, cv2.ROTATE_90_CLOCKWISE)
                    img2_rotated = cv2.rotate(img2_resized, cv2.ROTATE_90_CLOCKWISE)

                    # Display the resized frames for live viewing
                    #cv2.imshow('Cameras 1 & 2', np.hstack((img1_rotated, img2_rotated)))
                    #cv2.waitKey(1)  # Update the OpenCV window

                    # Save the original frames to videos
                    self.video_writer1.write(img1_rotated)
                    self.video_writer2.write(img2_rotated)

                    frame_count += 1  # Increment frame count here

                grabResult1.Release()
                grabResult2.Release()

                # Check if the recording duration has elapsed
                elapsed_time = time.time() - start_time
                print("ELAPSED TIME:", elapsed_time)
                if not ttl_signal_sent and elapsed_time >= 2:
                    self.send_ttl_signal_threaded(66,1000)  # Replace with the desired TTL value
                    ttl_signal_sent = True

            self.close_cv2_windows()

            # Stop recording after the loop
            self.stop_record()

            print("Frame count del ultimo video grabado: ", frame_count)

        except Exception as e:
            print(f"Error while start_record: {e}")

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
    def set_frame_rate_for_cameras(self, frame_rate):
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
            print("Acquisition frame rate cam 1: ", self.camera1.AcquisitionFrameRate.GetValue())
            print("Acquisition frame rate cam 2: ", self.camera2.AcquisitionFrameRate.GetValue())
            # Set the frame rate
            acquisition_frame_rate1.SetValue(frame_rate)
            acquisition_frame_rate2.SetValue(frame_rate)

            print(f"Frame rate set to: {frame_rate} FPS")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Close the cameras
            self.camera1.Close()
            self.camera2.Close()


if __name__ == "__main__":
    print(cv2.cuda.getCudaEnabledDeviceCount())

    save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
    subject_id = 'prueba_script'
    stimulation_pattern = ''

    # Create an instance of CameraControl
    camera_controller = CameraControl()
    print("Camera controller defined")
    # Initialize the cameras
    if camera_controller.init_cameras():
        try:
            camera_controller.check_frame_rate_for_cameras()
            camera_controller.set_frame_rate_for_cameras(120)

            
            for _ in range(1):  # Repeat the functioning
                # Start recording for each trial with different parameters
                recording_duration = 5  # Total recording duration (1 second for TTL + 3 seconds more)

                camera_controller.start_record(save_dir, subject_id, stimulation_pattern,recording_duration,frame_rate=120)                
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