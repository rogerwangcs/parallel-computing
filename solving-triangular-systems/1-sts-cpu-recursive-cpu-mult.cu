/*--------------------------------------------------------------
 *    Lower Triangular Matrix Inverse
 *    CPU RECURSION, CPU MAT MULT
 *    Run: nvcc --run helpers.cu check_solution.cu 1-sts-cpu-recursive-cpu-mult.cu
 *    Author: Roger Wang
 *-------------------------------------------------------------- */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "check_solution.h"
#include "helpers.h"
#include "timerc.h"

// Recursively invert A11, A22, and then A21
void inverseRecurse(double *a, int idx, int size, int n) {
    // Base Case: Invert if a is a simple 2x2 matrix
    if (size == 2) {
        invertTwoByTwo(a, idx, n);
        return;
    }

    // Calculate starting index for the 3 submatrices and make recursive calls
    int a11_idx = idx;                            // upper left submatrix
    int a22_idx = idx + size / 2 * n + size / 2;  // lower right submatrix
    int a21_idx = idx + size / 2 * n;             // bottom left full submatrix
    inverseRecurse(a, idx, size / 2, n);          // recurse on upper left submatrix
    inverseRecurse(a, a22_idx, size / 2, n);      // recurse on bottom right submatrix

    // Invert A21
    double *res = (double *)malloc(size * size * sizeof(double));
    multSubMat(a, a22_idx, a21_idx, size / 2, n, res);  // A22 * A21
    copyMat(res, a, a21_idx, size / 2, n);              // Put result into A21
    multSubMat(a, a21_idx, a11_idx, size / 2, n, res);  // A21 * A11
    copyMat(res, a, a21_idx, size / 2, n);              // Put result into A21
    addNegative(a, a21_idx, size / 2, n);               // Add negative sign to A21

    free(res);
    return;
}

// Just provide array input size (power of 2)
int cpu_recursive_cpu_mult(int inputSize, int check, int debug) {
    printf("\nCPU Recursive, CPU Multiplication\n");
    // Initialize
    float cpu_time;
    int n = pow(2, inputSize);  // Matrix size: 2^x = n
    double *a, *a_old, *b, *b_old;
    a = initMat(n);
    a_old = initMat(n);
    copyMat(a_old, a, 0, n, n);
    b = initVec(n);
    b_old = initVec(n);
    copyVec(b_old, b, n);

    // Begin Inversion
    cstart();
    inverseRecurse(a, 0, n, n);
    cend(&cpu_time);
    printf("Total Time: %f\n", cpu_time);
    // End Inversion

    // Check Solution
    if (check)
        checkSolution(a, a_old, b_old, n, debug);

    free(a);
    free(a_old);
    free(b);
    free(b_old);
    return 0;
}
