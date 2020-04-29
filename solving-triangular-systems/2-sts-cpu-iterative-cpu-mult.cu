#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../timerc.h"

/*--------------------------------------------------------------
 *    Lower Triangular Matrix Inverse
 *    CPU Iterative, CPU MAT MULT
 *    Solves Lower Triangle System of Equations

 *    Author: Roger Wang
 *-------------------------------------------------------------- */

void invertHelper(double *a, int idx, int size, int n) {
    // Base Case: Invert if a is a simple 2x2 matrix
    if (size == 2) {
        invertTwoByTwo(a, idx, n);
        return;
    }

    // Calculate starting index for the 3 submatrices
    int a11_idx = idx;                            // upper left submatrix
    int a22_idx = idx + size / 2 * n + size / 2;  // lower right submatrix
    int a21_idx = idx + size / 2 * n;             // bottom left full submatrix

    // Invert A21
    double *res = (double *)malloc(size * size * sizeof(double));
    multSubMat(a, a22_idx, a21_idx, size / 2, n, res);  // A22 * A21
    copyMat(res, a, a21_idx, size / 2, n);              // Put result into A21
    multSubMat(a, a21_idx, a11_idx, size / 2, n, res);  // A21 * A11
    copyMat(res, a, a21_idx, size / 2, n);              // Put result into A21
    addNegative(a, a21_idx, size / 2, n);               // Add negative sign to A21
    free(res);
}

void invertBottomUp(double *a, int n) {
    int total_iter = log2(n);
    for (int iter = 0; iter < total_iter; iter++) {
        int total_ops = pow(2, total_iter - iter - 1);
        for (int op_idx = 0; op_idx < total_ops; op_idx++) {
            int block_size = n / total_ops;
            int inv_idx = op_idx * ((n * n / total_ops) + block_size);
            // printf("Inv_idx: %d Block Size: %d\n", inv_idx, block_size);
            invertHelper(a, inv_idx, block_size, n);
        }
    }
}

// Just provide array input size (power of 2)
int cpu_iterative_cpu_mult(int inputSize, int check, int debug) {
    printf("\nCPU Iterative, CPU Multiplication\n");
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
    invertBottomUp(a, n);
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