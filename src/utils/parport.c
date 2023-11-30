#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/io.h>
#include <fcntl.h>

#define BASE_ADDRESS 0x3100

// To compile: gcc -O parport.c -o parport
// After compiling, set suid: chmod +s parport then, copy to /usr/sbin/

int main(int argc, char *argv[]) {
    // Check for correct number of arguments
    if (argc != 3) {
        printf("Usage: %s <value> <duration>\n", argv[0]);
        return 1;
    }

    // Parse value and duration from arguments
    int value = atoi(argv[1]);
    long duration = strtol(argv[2], NULL, 10);

    printf("Value: %d, Duration: %ld ms\n", value, duration);

    // Request I/O permissions for the parallel port
    if (ioperm(BASE_ADDRESS, 3, 1) == -1) {
        perror("Couldn't request I/O permissions");
        exit(1);
    }

    // Checkpoint 1: Permissions granted
    printf("I/O permissions granted.\n");

    // Set pins of the parallel port according to the given value
    outb(value, BASE_ADDRESS);
    printf("Pins set to value: %d.\n", value);

    // Sleep for the specified duration
    usleep(duration * 1000);  // Convert duration from ms to microseconds

    // Set all pins of the parallel port to low (logic 0)
    outb(0, BASE_ADDRESS);

    // Checkpoint 2: Pins set to low
    printf("All pins set to low.\n");

    // Release I/O permissions for the parallel port
    ioperm(BASE_ADDRESS, 3, 0);

    // Checkpoint 3: Permissions released
    printf("I/O permissions released.\n");

    return 0;
}

