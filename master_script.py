import threading
from modules import ttl_signals, camera_control, led_pattern, data_logging
from time import sleep
import random

def main(frame_rate):
    # Initialization code here

    for i in range(7):  # 7 Trials
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
