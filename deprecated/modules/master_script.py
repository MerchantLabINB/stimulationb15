import threading
from modules import camera_control, led_pattern, data_logging, ttl_signals
from time import sleep
import random
import numpy as np
import configparser  # Add this import for handling INI files
import subprocess
# Function to load configuration from an INI file
def load_config():
    config = configparser.ConfigParser()
    config.read('config/config.ini')  # Specify the path to your INI file
    return config

def main():
    config = load_config()  # Load configuration from the INI file

    # Call the GUI interface script using subprocess
    subprocess.Popen(["python", "gui_interface_script.py"])

    def randperm(n):
        return np.random.permutation(n) + 1

    # Example usage:
    n = int(config['Experiment']['num_trials'])  # Read the number of trials from the INI file
    permutation = randperm(n)
    print(permutation)

    # Initialization code here

    for i in range(n):  # n Trials
        # Random Wait
        wait_time = random.uniform(4, 8)
        sleep(wait_time)

        # Initialize Video
        threading.Thread(target=camera_control.init_record).start()

        # 1-sec Wait
        sleep(1)

        # Start TTL Signal and LED Pattern
        ttl_thread = threading.Thread(target=ttl_signals.send_signal)
        led_thread = threading.Thread(target=led_pattern.start_pattern)
        ttl_thread.start()
        led_thread.start()
        ttl_thread.join()
        led_thread.join()

        # 3-sec Wait
        sleep(3)

        # Stop Recording
        camera_control.stop_record()

    # Log data to CSV
    data_logging.save_to_csv()

if __name__ == "__main__":
    main()

