include <stdio.h>

// Declare the function defined in myfunction.c
extern void parallel();

int main() {
    printf("Calling myFunction...\n");
    
    // Call your function
    parallel();
    
    printf("myFunction called!\n");
    
    return 0;
}