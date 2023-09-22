""""
import parallel
p = parallel.Parallel('/dev/parport0')
print(p)

import os

# Open the parallel port for writing
parallel_port = os.open("/dev/parport0", os.O_WRONLY)

# Send a byte of data (e.g., 0xFF) to the parallel port
os.read(parallel_port,255)

# Close the parallel port
os.close(parallel_port)
"""

from expyriment import io

pports = [io.ParallelPort('/dev/' + pp)
          for pp in io.ParallelPort.get_available_ports()]

prev_state = []
while True:
  state = str([bin(p.poll()) for p in pports])
  if state != prev_state:
      print(state)
      prev_state = state

"""
import sys

# Print sys.path before appending
print("Before appending:")
print(sys.path)

# Append new path
sys.path.append("/anaconda3/envs/pylon_env/lib/python3.8/site-packages/expyriment/expyriment")

# Print sys.path after appending
print("After appending:")
print("PATHHH",sys.path)

from expyriment import io
print("Expyriment imported successfully.")
import parallel

# Open the parallel port as a serial port
port = parallel.Parallel()
# Send data to the parallel port (use binary data)
port.write('255')

# Close the port when do
port.close()
"""

"""from psychopy import parallel
port = parallel.ParallelPort(address=u'0xB010')"""
