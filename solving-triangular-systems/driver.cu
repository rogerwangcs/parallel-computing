#include <stdio.h>
#include <stdlib.h>

// Load Matrix Inversion Algorithms
#include "1-sts-cpu-recursive-cpu-mult.cu"
#include "2-sts-cpu-iterative-cpu-mult.cu"
// #include "3-sts-cpu-recursive-gpu-mult.cu"
// #include "4-sts-cpu-iterative-gpu-mult.cu"
// #include "5-sts-gpu-iterative-cpu-mult.cu"
// #include "6-sts-gpu-iterative-gpu-mult.cu"

int main(int argc, char *argv[]) {
    int n = 4;      // input matrix size
    int check = 0;  // check algorithm output
    int debug = 1;  // print output

    if (argv[1] != NULL)  // set input
        n = (int)strtol(argv[1], NULL, 12);
    if (argv[2] != NULL)
        check = (int)strtol(argv[2], NULL, 12);
    if (argv[3] != NULL)
        debug = (int)strtol(argv[3], NULL, 12);

    printf("Matrix Size: %d, Check Results?: %d, Print Output?: %d\n", n, check, debug);

    // Run Algorithms
    cpu_recursive_cpu_mult(n, check, debug);
    cpu_iterative_cpu_mult(n, check, debug);
    return 0;
}