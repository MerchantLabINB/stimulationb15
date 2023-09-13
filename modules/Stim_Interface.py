import tkinter as tk
from tkinter import ttk
import random
import time

import configparser

# Use current CPU time as seed
random.seed(time.process_time())

def save_configuration():
    config['GUI'] = {
        'PatternTime': pattern_time_var.get(),
        'IntertrialTime': intertrial_time_var.get(),
        'HoldingTime': '1.0',  # Fixed value
        'FrameRate': '120',    # Fixed value
        'EvocatedTime': '0.0'  # Fixed value
    }    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

def load_configuration():
    try:
        config.read('config.ini')
        pattern_time_var.set(config['GUI']['PatternTime'])
        intertrial_time_var.set(config['GUI']['IntertrialTime'])
    except FileNotFoundError:
        pass  # Handle the case where the config file doesn't exist


def start_button_click():

    frame_rate = float(frame_rate_entry.get())
    holding_time = float(holding_time_entry.get())
    pattern_time = float(pattern_time_entry.get())
    evocated_time = float(evocated_time_entry.get())
    intertrial_time = random.uniform(5, 7)  # Pseudo-random between 5 to 7 seconds
    intertrial_time_var.set(f"{intertrial_time:.2f}")
    # Your code to start recording or similar actions
    save_configuration():
def stop_button_click():
    # Your code to stop recording or similar actions
    print('Stopping')



root = tk.Tk()
root.title("GUI Interface")

frame_rate_var = tk.StringVar(value="120")
frame_rate_label = ttk.Label(root, text="Frame Rate:")
frame_rate_label.pack()
frame_rate_entry = ttk.Entry(root, textvariable=frame_rate_var)
frame_rate_entry.pack()

holding_time_var = tk.StringVar(value="1.0")
holding_time_label = ttk.Label(root, text="Holding Time:")
holding_time_label.pack()
holding_time_entry = ttk.Entry(root, textvariable=holding_time_var)
holding_time_entry.pack()

pattern_time_var = tk.StringVar()
pattern_time_label = ttk.Label(root, text="Pattern Time:")
pattern_time_label.pack()
pattern_time_entry = ttk.Entry(root, textvariable=pattern_time_var)
pattern_time_entry.pack()

evocated_time_var = tk.StringVar()
evocated_time_label = ttk.Label(root, text="Evocated Time:")
evocated_time_label.pack()
evocated_time_entry = ttk.Entry(root, textvariable=evocated_time_var)
evocated_time_entry.pack()

intertrial_time_var = tk.StringVar()
intertrial_time_label = ttk.Label(root, text="Intertrial Time:")
intertrial_time_label.pack()
intertrial_time_entry = ttk.Entry(root, textvariable=intertrial_time_var)
intertrial_time_entry.pack()

start_button = ttk.Button(root, text="Start", command=start_button_click)
start_button.pack()

stop_button = ttk.Button(root, text="Stop", command=stop_button_click, command=save_configuration)
stop_button.pack()


# Initialize a configuration file
config = configparser.ConfigParser()

# Function to save the configuration



root.mainloop()
