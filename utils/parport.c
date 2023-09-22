#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/io.h>
#include <fcntl.h>

#define BASE_ADDRESS 0x3100

// To compile:  gcc -O parport.c -o parport
// After compiling, set suid:  chmod +s parport   then, copy to /usr/sbin/

int main(int argc, char *argv[]) {
 if (argc != 2) {
    printf("Usage: %s <value>\n", argv[0]);
    return 1;
  }

  int value = atoi(argv[1]);
  
  // Request I/O permissions for the parallel port
  if (ioperm(BASE_ADDRESS, 3, 1) == -1) {
    perror("Couldn't request I/O permissions");
    exit(1);
  }

  // Checkpoint 1: Permissions granted
  printf("I/O permissions granted.\n");

  // Set all pins of the parallel port to high (logic 1)
  outb(255, BASE_ADDRESS);  // 255 in binary is 11111111, setting all pins to high
  sleep(1);  // Sleep for 1 second

  // Checkpoint 2: Pins set to high
  printf("All pins set to high.\n");

  // Set all pins of the parallel port to low (logic 0)
  outb(0, BASE_ADDRESS);  // 0 in binary is 00000000, setting all pins to low

  // Checkpoint 3: Pins set to low
  printf("All pins set to low.\n");

  // Release I/O permissions for the parallel port
  ioperm(BASE_ADDRESS, 3, 0);

  // Checkpoint 4: Permissions released
  printf("I/O permissions released.\n");

  return 0;
}



