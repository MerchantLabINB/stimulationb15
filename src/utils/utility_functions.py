import os
import subprocess
import configparser

def run_c_program_with_value(value):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(value)
    print(script_dir)
    # Define the relative path to the parport executable
    parport_path = os.path.join(script_dir, "parport")
    print(parport_path)
    # Define the command to run your C program with the provided value
    command = f"sudo {parport_path} {value}"

    # Execute the command and capture both stdout and stderr
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    # Check if there was an error
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None

    # Return the output of your C program
    return stdout
# src/utils/utility_functions.py



def load_configuration(config, pattern_time_var, intertrial_time_var):
    try:
        config.read('config/config.ini')
        # Populate the configuration settings here as needed
        pattern_time_var.set(config['GUI']['PatternTime'])
        intertrial_time_var.set(config['GUI']['IntertrialTime'])

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
    
def check_frame_rate_for_cameras():
    from pypylon import pylon

    try:
        # Initialize the first camera
        camera1 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera1.Open()
        node_map1 = camera1.GetNodeMap()
        acquisition_frame_rate1 = node_map1.GetNode("AcquisitionFrameRate")
        current_frame_rate1 = acquisition_frame_rate1.GetValue()

        # Try to initialize the second camera (if available)
        try:
            camera2 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDeviceByIndex(1))
            camera2.Open()
            node_map2 = camera2.GetNodeMap()
            acquisition_frame_rate2 = node_map2.GetNode("AcquisitionFrameRate")
            current_frame_rate2 = acquisition_frame_rate2.GetValue()
            print(f"Camera 2 Frame Rate: {current_frame_rate2} FPS")
        except Exception as e:
            print("Camera 2 not found or could not be initialized.")

        print(f"Camera 1 Frame Rate: {current_frame_rate1} FPS")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the cameras
        camera1.Close()
        if camera2.IsOpen():
            camera2.Close()




# Example usage:
if __name__ == "__main__":
    # Probando el envio de señal TTL
    value_to_pass = "9"  # Replace with the desired value
    output = run_c_program_with_value(value_to_pass)
    if output is not None:
        print("Output:", output)

    # Call the function to check frame rates for both cameras
    check_frame_rate_for_cameras()