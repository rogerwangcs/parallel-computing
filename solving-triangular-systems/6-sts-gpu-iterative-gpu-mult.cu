#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "check_solution.h"
#include "helpers.h"
#include "timerc.h"

/*--------------------------------------------------------------
 *    Lower Triangular Matrix Inverse
 *    GPU Iterative, GPU MAT MULT
 *    Run: Driver ( with -arch=sm_35 -rdc=true)
 *    Author: Roger Wang
 *-------------------------------------------------------------- */

// updated to be called from within iteration kernel (dynamic parallelism)
__device__ void parallelMultMat2(double *a, int idx_1, int idx_2, int size, int n, double *res, int copyToFirst) {
    // device variables
    double *a1_d, *a2_d, *res_d;
    cudaMalloc((void **)&a1_d, size * size * sizeof(double));
    cudaMalloc((void **)&a2_d, size * size * sizeof(double));
    cudaMalloc((void **)&res_d, size * size * sizeof(double));
    cudaMemcpy2DAsync(a1_d, size * sizeof(double),
                      a + idx_1, n * sizeof(double),
                      size * sizeof(double),
                      size, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DAsync(a2_d, size * sizeof(double),
                      a + idx_2, n * sizeof(double),
                      size * sizeof(double),
                      size, cudaMemcpyDeviceToDevice);

    // execute kernel
    multMat_kernel<<<size, size>>>(a1_d, a2_d, size, res_d);
    cudaDeviceSynchronize();

    // copy and return calculations
    if (copyToFirst == 0) {
        cudaMemcpy2DAsync(a + idx_1, n * sizeof(double),
                          res_d, size * sizeof(double),
                          size * sizeof(double),
                          size, cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy2DAsync(a + idx_2, n * sizeof(double),
                          res_d, size * sizeof(double),
                          size * sizeof(double),
                          size, cudaMemcpyDeviceToDevice);
    }

    // printSubMat(a, idx_2, size, n, "Result");
    // free device memory
    cudaFree(a1_d);
    cudaFree(a2_d);
    cudaFree(res_d);
    return;
}

__device__ void invertHelperParallelMultAndIter(double *a_d, int idx, int size, int n) {
    // Base Case: Invert if a is a simple 2x2 matrix
    if (size == 2) {
        invertTwoByTwo(a_d, idx, n);
        return;
    }

    // Calculate starting index for the 3 submatrices
    int a11_idx = idx;                            // upper left submatrix
    int a22_idx = idx + size / 2 * n + size / 2;  // lower right submatrix
    int a21_idx = idx + size / 2 * n;             // bottom left full submatrix

    // Invert A21
    double *res_d;
    cudaMalloc((void **)&res_d, size * size * sizeof(double));

    // Invert A21
    double *res = (double *)malloc(size * size * sizeof(double));
    parallelMultMat2(a_d, a22_idx, a21_idx, size / 2, n, res, 1);  // A22 * A21 & put result into A21
    parallelMultMat2(a_d, a21_idx, a11_idx, size / 2, n, res, 0);  // A21 * A11 & put result into A21
    addNegative(a_d, a21_idx, size / 2, n);                        // Add negative sign to A21
    cudaFree(res_d);
}

__global__ void invertPass_kernel2(double *a_d, int total_ops, int n) {
    int block_size = n / total_ops;
    int inv_idx = threadIdx.x * ((n * n / total_ops) + block_size);
    // printf("Inv_idx: %d Block Size: %d\n", inv_idx, block_size);
    invertHelperParallelMultAndIter(a_d, inv_idx, block_size, n);
}

void invertBottomUp2(double *a_d, int n) {
    int total_iter = log2(n);
    for (int iter = 0; iter < total_iter; iter++) {
        int total_ops = pow(2, total_iter - iter - 1);
        invertPass_kernel2<<<1, total_ops>>>(a_d, total_ops, n);
        cudaDeviceSynchronize();
    }
}

int gpu_iterative_gpu_mult(int inputSize, int check, int debug) {
    printf("\nGPU Iterative, GPU Multiplication\n");
    // Initialize
    int n = pow(2, inputSize);  // Matrix size: 2^x = n
    double *a, *a_old, *b, *b_old;
    a = initMat(n);
    a_old = initMat(n);
    copyMat(a_old, a, 0, n, n);
    b = initVec(n);
    b_old = initVec(n);
    copyVec(b_old, b, n);

    // Start Inversion
    float cpu_time, cpu_time2;
    float gpu_time;
    cstart();
    double *a_d;
    cudaMalloc((void **)&a_d, n * n * sizeof(double));
    cudaMemcpy(a_d, a, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cend(&cpu_time);
    gstart();
    invertBottomUp2(a_d, n);
    gend(&gpu_time);
    cstart();
    cudaMemcpy(a, a_d, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cend(&cpu_time2);
    cudaFree(a_d);
    printf("Total Time: %f\n", cpu_time + cpu_time2 + gpu_time);

    // Check Solution
    if (check)
        checkSolution(a, a_old, b_old, n, debug);

    free(a);
    free(a_old);
    free(b);
    free(b_old);
    return 0;
}
