"""
import parallel

# Find the first parallel port (usually "/dev/parport0")
port = parallel.Parallel()

# Check if the port is open and available
if port.isPrinterBusy():
    print("Parallel port is busy.")
else:
    # Define the data to send (8 bits)
    data_to_send = 0b10101010

    # Send data to the data port
    port.setData(data_to_send)

    # Close the parallel port
    port.close()
"""

# ttl_signals.py

from time import sleep
import random
from expyriment.io import ParallelPort

# Initialize the Parallel Port; replace '/dev/parport0' with the actual device path on your machine
pp = ParallelPort('/dev/parport0')

def generate_pseudo_random_time(lower=5, upper=7):
    return random.uniform(lower, upper)

def send_ttl_signal(signal_code):
    """
    Sends a TTL signal with the given code using expyriment's ParallelPort.
    """
    pp.set_data(signal_code)
    sleep(0.002)  # 2ms pulse
    pp.set_data(0)  # Reset to 0

def initiate_trial():
    intertrial_time = generate_pseudo_random_time()
    holding_time = 1.0  # in seconds

    # Start of Trial
    send_ttl_signal(255)  # You can adjust this code as needed

    # Holding Time
    sleep(holding_time)

    # Inter-trial Time
    sleep(intertrial_time)

    # End of Trial
    send_ttl_signal(0)  # You can adjust this code as needed

    return intertrial_time

if __name__ == "__main__":
    # Test code for the ttl_signals.py module
    random.seed(None)  # Seed with current system time for more randomness

    print("Starting test trials...")
    for i in range(3):
        intertrial_time = initiate_trial()
        print(f"Trial {i+1} completed with intertrial time: {intertrial_time} seconds")
    print("All test trials completed.")
