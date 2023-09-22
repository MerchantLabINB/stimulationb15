import threading
import tkinter as tk
from tkinter import ttk
import random
import time
import configparser
from src.hardware import camera_control

# Definiendo variables importantes
# Donde se graban las cosas
save_dir = '/home/brunobustos96/Documents/stimulationB15/data/'
subject_id = 'probando desde GUI'
stimulation_pattern = '' #eSTO SE DEBE ACTUALIZAR DESDE EL CONFIG Y DESDE LOS CSV
camera_thread = None
video_writer1 = None
video_writer2 = None
# Function to initialize cameras
camera1, camera2 = camera_control.init_cameras()

# Use current CPU time as seed
random.seed(time.process_time())

# Initialize the main Tkinter root window
root = tk.Tk()
root.title("GUI Interface")

# Define global variables (StringVar variables and config)
frame_rate_var = tk.StringVar(value="120")
holding_time_var = tk.StringVar(value="1.0")
pattern_time_var = tk.StringVar()
evocated_time_var = tk.StringVar()
intertrial_time_var = tk.StringVar()
config = configparser.ConfigParser()

# Function to load configuration
def load_configuration():
    try:
        config.read('config/config.ini')
        pattern_time_var.set(config['GUI']['PatternTime'])
        intertrial_time_var.set(config['GUI']['IntertrialTime'])
    except FileNotFoundError:
        print("No se encontró el archivo de configuración.")
        pass

# Function to save configuration
def save_configuration():
    config['GUI'] = {
        'PatternTime': pattern_time_var.get(),
        'IntertrialTime': intertrial_time_var.get(),
        'HoldingTime': holding_time_var.get(),
        'FrameRate': frame_rate_var.get(),
        'EvocatedTime': evocated_time_var.get()
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Function to start trials


# Remove the global video_writer1 and video_writer2 variables

# Function to start the camera recording thread
def start_camera():
    global camera_thread
    camera_thread = threading.Thread(target=camera_control.start_record, args=(camera1, camera2, save_dir, subject_id, stimulation_pattern))
    camera_thread.start()

# Function to stop the camera recording thread
def stop_camera():
    global camera_thread
    if camera_thread:
        camera_control.stop_record(camera1, camera2, video_writer1, video_writer2)
        camera_thread.join()  # Wait for the camera thread to finish
        camera_thread = None



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

start_button = ttk.Button(root, text="Start Trials", command=start_camera)
start_button.pack()

stop_button = ttk.Button(root, text="Stop Trials", command=stop_camera)
stop_button.pack()


# Load configuration if it exists
load_configuration()

# Start the GUI main loop
root.mainloop()
