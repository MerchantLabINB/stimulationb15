# gui_interface.py
import threading
import tkinter as tk
from tkinter import ttk
import random
import time
import configparser
import os
import numpy as np
import cv2
from queue import Queue
from src.hardware.camera_control import CameraControl  # Import the CameraControl class using a relative import
from src.utils.utility_functions import load_configuration, save_configuration
from pypylon import pylon

# Define global variables
save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
subject_id = 'probando desde GUI'
stimulation_pattern = ''
camera_controller = None
recording_flag = False
elapsed_time = 0

def video_processing_thread(frame_queue):
    global recording_flag

    while recording_flag:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Process the frame here (e.g., display, save, send TTL, etc.)

# Add a variable to keep track of elapsed time
elapsed_time = 0

# Modify the start_trials function
def start_trials(frame_queue):
    global recording_flag, elapsed_time, start_time

    if camera_controller:
        recording_flag = True
        try:
            recording_duration = float(recording_duration_var.get())
            print("Recording duration:", recording_duration)
            for _ in range(3):  # Repeat the functioning 3 times (adjust as needed)
                start_time_recording = time.time()
                camera_controller.start_record(save_dir, subject_id, stimulation_pattern, recording_duration,
                                               frame_rate=int(frame_rate_var.get()))
                print("Recording started...")
                print("start_record tardó ",time.time()-start_time_recording)

                # Define start_time just before the loop starts
                start_time = time.time()

                # Reset elapsed time for each trial
                elapsed_time = 0

                while elapsed_time < recording_duration:
                    # Update elapsed time
                    elapsed_time = time.time() - start_time
                    print("ELAPSED TIME:", elapsed_time)
                    
                    # Check if elapsed time is within the specified time range (1 second to 1.1 seconds) and send TTL signal
                    if 1 <= int(elapsed_time) <= 1.1:
                        value_to_pass = "66"  # Replace with the desired value
                        camera_controller.send_ttl_signal(value_to_pass)
                        break  # Exit the loop immediately after sending TTL signal

                    # Sleep for a short duration to avoid busy-waiting
                    time.sleep(0.01)

                
                # Stop recording for the current trial
                camera_controller.stop_record()
                print("Recording stopped.")

                intertrial_time = float(intertrial_time_var.get())
                time.sleep(intertrial_time)

                camera_controller.close_cv2_windows()

                if not recording_flag:
                    break  # Exit the loop if recording_flag is False

            print("Program executed successfully")
        except ValueError:
            print("Invalid recording duration or intertrial time entered.")
        except KeyboardInterrupt:
            pass
    else:
        print("Camera controller is not initialized.")



def stop_trials():
    global recording_flag
    print("Stop_trials has been activated")
    recording_flag = False
    # Stop the recording thread gracefully
    camera_controller.stop_record()


if __name__ == "__main__":
    # Initialize the main Tkinter root window
    root = tk.Tk()
    root.title("GUI Interface")

    ttk.Style().configure("TButton", font=("TkDefaultFont", 16))
    ttk.Style().configure("TLabel", font=("TkDefaultFont", 16))
    ttk.Style().configure("TEntry", font=("TkDefaultFont", 16))

    # Create a queue to communicate between the main thread and the video processing thread
    frame_queue = Queue()

    # Create an instance of CameraControl and initialize the cameras
    camera_controller = CameraControl()

    if camera_controller.init_cameras():
        try:
            frame_rate_var = tk.StringVar(value="120")
            holding_time_var = tk.StringVar(value="1.0")
            pattern_time_var = tk.StringVar()
            evocated_time_var = tk.StringVar()
            intertrial_time_var = tk.StringVar()
            recording_duration_var = tk.StringVar()
            config = configparser.ConfigParser()

            load_configuration(
                config,
                pattern_time_var,
                intertrial_time_var,
                frame_rate_var,
                holding_time_var,
                evocated_time_var,
                recording_duration_var
            )

            frame_rate_label = ttk.Label(root, text="Frame Rate:")
            frame_rate_label.pack()
            frame_rate_entry = ttk.Entry(root, textvariable=frame_rate_var)
            frame_rate_entry.pack()

            holding_time_label = ttk.Label(root, text="Holding Time:")
            holding_time_label.pack()
            holding_time_entry = ttk.Entry(root, textvariable=holding_time_var)
            holding_time_entry.pack()

            pattern_time_label = ttk.Label(root, text="Pattern Time:")
            pattern_time_label.pack()
            pattern_time_entry = ttk.Entry(root, textvariable=pattern_time_var)
            pattern_time_entry.pack()

            evocated_time_label = ttk.Label(root, text="Evocated Time:")
            evocated_time_label.pack()
            evocated_time_entry = ttk.Entry(root, textvariable=evocated_time_var)
            evocated_time_entry.pack()

            intertrial_time_label = ttk.Label(root, text="Intertrial Time:")
            intertrial_time_label.pack()
            intertrial_time_entry = ttk.Entry(root, textvariable=intertrial_time_var)
            intertrial_time_entry.pack()

            recording_duration_label = ttk.Label(root, text="Recording Duration:")
            recording_duration_label.pack()
            recording_duration_entry = ttk.Entry(root, textvariable=recording_duration_var)
            recording_duration_entry.pack()

            start_button = ttk.Button(root, text="Start Trials", command=lambda: start_trials(frame_queue))
            start_button.pack()

            stop_button = ttk.Button(root, text="Stop Trials", command=stop_trials)
            stop_button.pack()

            save_button = ttk.Button(root, text="Save Configuration", command=lambda: save_configuration(
                config,
                pattern_time_var.get(),
                intertrial_time_var.get(),
                holding_time_var.get(),
                frame_rate_var.get(),
                evocated_time_var.get(),
                recording_duration_var.get()
            ))
            save_button.pack()

            # Start the video processing thread
            video_thread = threading.Thread(target=video_processing_thread, args=(frame_queue,))
            video_thread.daemon = True
            video_thread.start()

            # Start the GUI main loop
            root.mainloop()
        except KeyboardInterrupt:
            # Exit the loop when the user presses Ctrl+C
            pass
    else:
        print("Failed to initialize cameras.")

