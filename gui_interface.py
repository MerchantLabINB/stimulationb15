import threading
import tkinter as tk
from tkinter import ttk
import random
import time
import configparser
from src.hardware.camera_control import CameraControl  # Import the CameraControl class using a relative import
from src.utils.utility_functions import load_configuration, save_configuration
from pypylon import pylon

# Definiendo variables importantes
# Donde se graban las cosas
save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
subject_id = 'probando desde GUI'
stimulation_pattern = ''  # Esto se debe actualizar desde el config y desde los CSV
camera_thread = None
camera_controller = None

# Initialize the main Tkinter root window
root = tk.Tk()
root.title("GUI Interface")

# Define global variables (StringVar variables and config)
frame_rate_var = tk.StringVar(value="120")
holding_time_var = tk.StringVar(value="1.0")
pattern_time_var = tk.StringVar()
evocated_time_var = tk.StringVar()
intertrial_time_var = tk.StringVar()
recording_duration_var = tk.StringVar()
config = configparser.ConfigParser()

# Initialize recording duration as a global variable
recording_duration = 3  # Default value (adjust as needed)

# Load configuration settings from the file and pass the variables
load_configuration(config, pattern_time_var, intertrial_time_var)

recording_flag = False

def start_trials():
    global camera_thread, recording_flag, recording_duration

    if camera_controller:
        recording_flag = True
        try:
            recording_duration = float(recording_duration_var.get())  # Get the recording duration from the entry field

            for _ in range(3):  # Repeat the functioning 3 times (adjust as needed)
                # Start recording for each trial with different parameters
                camera_controller.start_record(save_dir, subject_id, stimulation_pattern, recording_duration)

                # Stop recording for the current trial
                camera_controller.stop_record()

                # Add intertrial time here (integer seconds)
                intertrial_time = float(intertrial_time_var.get())  # Get the intertrial time from the entry field
                time.sleep(intertrial_time)

                camera_controller.close_cv2_windows()
            print("Program executed successfully")
        except ValueError:
            print("Invalid recording duration or intertrial time entered.")
        except KeyboardInterrupt:
            # Exit the loop when the user presses Ctrl+C
            pass
    else:
        print("Camera controller is not initialized.")

# Function to stop trials
def stop_trials():
    global recording_flag
    print("Stop_trials has been activated")
    recording_flag = False  # Set the flag to stop the recording

# Create an instance of CameraControl and initialize the cameras
camera_controller = CameraControl()  # Make sure CameraControl is imported

if camera_controller.init_cameras():
    # Create and pack GUI elements
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

    # You can also add a label and entry for recording duration similarly
    recording_duration_label = ttk.Label(root, text="Recording Duration:")
    recording_duration_label.pack()
    recording_duration_entry = ttk.Entry(root, textvariable=recording_duration_var)
    recording_duration_entry.pack()

    start_button = ttk.Button(root, text="Start Trials", command=start_trials)
    start_button.pack()

    stop_button = ttk.Button(root, text="Stop Trials", command=stop_trials)
    stop_button.pack()

    # Create a "Save" button in your GUI
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

    # Start the GUI main loop
    root.mainloop()
else:
    print("Failed to initialize cameras.")
