import subprocess

def run_parport_program():
    try:
        # Run the parport C program using subprocess
        #subprocess.run(["ls"], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        subprocess.run(["sudo", "./parport"], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        print("C program executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.returncode}\n{e.stderr}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Call the Python function to run the C program
run_parport_program()
