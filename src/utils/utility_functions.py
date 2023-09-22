import os
import subprocess

def run_c_program_with_value(value):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
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

# Example usage:
if __name__ == "__main__":
    value_to_pass = "255"  # Replace with the desired value
    output = run_c_program_with_value(value_to_pass)
    if output is not None:
        print("Output:", output)