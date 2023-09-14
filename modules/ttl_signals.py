from expyriment.io import ParallelPort
import random
from time import sleep

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

if __name__ == "__main__":
    try:
        # Define the data value (0-255), pulse duration (in milliseconds), and number of pulses
        data_value = 0xFF  # 0xFF represents all bits high (binary 11111111)
        pulse_duration_ms = 500  # 500 milliseconds (0.5 seconds)
        pulse_count = 5  # Number of pulses to send

        # Send the specified number of pulses
        for _ in range(pulse_count):
            send_ttl_signal(data_value)
            sleep(pulse_duration_ms / 1000.0)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
