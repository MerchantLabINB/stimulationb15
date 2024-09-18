import os
import subprocess
import configparser

def run_c_program(value, duration):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Print the provided value and duration for debugging purposes
    print(f"Value: {value}, Duration: {duration} ms")
    print(f"Script Directory: {script_dir}")

    # Define the relative path to the parport executable
    parport_path = os.path.join(script_dir, "parport")
    print(f"Parport Path: {parport_path}")

    # Define the command to run your C program with the provided value and duration
    command = f"sudo {parport_path} {value} {duration}"

    # Execute the command and capture both stdout and stderr
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    # Check if there was an error
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None

    # If there's output, print it (for debugging or confirmation)
    if stdout:
        print(stdout)

    # Return the output of your C program (or you could just return True to indicate success)
    return stdout
# src/utils/utility_functions.py



def load_configuration(config, pattern_time_var, intertrial_time_var, frame_rate_var, holding_time_var, evocated_time_var, recording_duration_var):
    try:
        config.read('config/config.ini')
        # Populate the configuration settings here as needed
        pattern_time_var.set(config['GUI']['patterntime'])
        intertrial_time_var.set(config['GUI']['intertrialtime'])
        frame_rate_var.set(config['GUI']['framerate'])
        holding_time_var.set(config['GUI']['holdingtime'])
        evocated_time_var.set(config['GUI']['evocatedtime'])
        recording_duration_var.set(config['GUI']['recordduration'])

        print("Load configuration done")
    except FileNotFoundError:
        print("No se encontró el archivo de configuración.")
        pass



# Function to save configuration
def save_configuration(config, pattern_time, intertrial_time, holding_time, frame_rate, evocated_time, recording_duration):
    config['GUI'] = {
        'PatternTime': pattern_time,
        'IntertrialTime': intertrial_time,
        'HoldingTime': holding_time,
        'FrameRate': frame_rate,
        'EvocatedTime': evocated_time,
        'RecordDuration': recording_duration
    }
    with open('config/config.ini', 'w') as configfile:
        config.write(configfile)
        print("Se sobreescribió config.ini con éxito")

if __name__ == "__main__":
    value = 66
    duration = 1000  # Duration in milliseconds
    output = run_c_program(value, duration)
    if output is not None:
        print("Command executed successfully:\n", output)