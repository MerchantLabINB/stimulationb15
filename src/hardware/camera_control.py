from pypylon import pylon
import cv2
import time
import os
import random
import datetime

class CameraControl:
    def __init__(self):
        self.camera1 = None
        self.camera2 = None
        self.video_writer1 = None
        self.video_writer2 = None
        self.converter1 = None
        self.converter2 = None

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

    def grab_and_write_frames(self, show_video=True):
        if not self.camera1 or not self.camera2:
            print("Cameras are not initialized.")
            return
        
        try:
            if not self.camera1.IsGrabbing() and not self.camera2.IsGrabbing():
                self.camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            grabResult1 = self.camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            grabResult2 = self.camera2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():
                img1 = self.converter1.Convert(grabResult1).GetArray()
                img2 = self.converter2.Convert(grabResult2).GetArray()
                
                self.video_writer1.write(img1)
                self.video_writer2.write(img2)
                
                if show_video:
                    cv2.imshow('Camera 1', img1)
                    cv2.imshow('Camera 2', img2)
                    
            grabResult1.Release()
            grabResult2.Release()
        except Exception as e:
            print(f"Error while grabbing and writing frames: {e}")

    def start_record(self, save_dir, subject_id, stimulation_pattern, recording_duration, frame_rate=120):
        if not self.camera1 or not self.camera2:
            print("Cameras are not initialized.")
            return

        try:
            # Initialize video writers using the mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG' or other codecs

            current_time = time.strftime("%Y%m%d-%H%M%S")

            file_name1 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam1.avi')
            file_name2 = os.path.join(save_dir, f'{subject_id}_{stimulation_pattern}_{current_time}_cam2.avi')

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

            start_time = datetime.datetime.now()

            while (self.camera1.IsGrabbing() and self.camera2.IsGrabbing() and
                (datetime.datetime.now() - start_time).total_seconds() < recording_duration):
                self.grab_and_write_frames(show_video=True)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Pressed 'q' within start_record loop")
                    break
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


if __name__ == "__main__":
    save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
    subject_id = 'prueba_script'
    stimulation_pattern = ''

    # Create an instance of CameraControl
    camera_controller = CameraControl()

    # Initialize the cameras
    if camera_controller.init_cameras():
        try:
            for _ in range(3):  # Repeat the functioning 7 times
                # Start recording for each trial with different parameters
                recording_duration = 3  # seconds (trial duration)

                camera_controller.start_record(save_dir, subject_id, stimulation_pattern,recording_duration)                

                # Stop recording for the current trial
                camera_controller.stop_record()

                # Add intertrial time here (randomly generated)
                intertrial_time = random.uniform(1.0, 5.0)  # Adjust the range as needed
                time.sleep(intertrial_time)
                camera_controller.close_cv2_windows()
            print("Programa ejecutado correctamente")
        except KeyboardInterrupt:
            # Exit the loop when the user presses Ctrl+C
            pass

