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
