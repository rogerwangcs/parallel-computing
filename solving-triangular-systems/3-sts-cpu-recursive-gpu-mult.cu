#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "check_solution.h"
#include "helpers.h"
#include "timerc.h"

/*--------------------------------------------------------------
 *    Lower Triangular Matrix Inverse
 *    CPU RECURSION, GPU MAT MULT
 *    Run: Driver
 *    Author: Roger Wang
 *-------------------------------------------------------------- */

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// device kernel function for multMat parallel
__global__ void multMat_kernel(double *a1, double *a2, int size, double *res) {
    int block_idx = gridDim.x * blockIdx.y + blockIdx.x + 1;
    int i = block_idx * threadIdx.x;
    int j = block_idx * threadIdx.y;
    res[i * size + j] = 0.0;
    for (int k = 0; k < size; k++) {
        res[i * size + j] += a1[i * size + k] * a2[k * size + j];
    }
    return;
}

void parallelMultMat(double *a, int idx_1, int idx_2, int size, int n, int copyToFirst) {
    // device variables
    double *a1_d, *a2_d, *res_d;
    gpuErrchk(cudaMallocPitch((void **)&a1_d, size * size * sizeof(double)));
    gpuErrchk(cudaMallocPitch((void **)&a2_d, size * size * sizeof(double)));
    gpuErrchk(cudaMallocPitch((void **)&res_d, size * size * sizeof(double)));
    gpuErrchk(cudaMemcpy2D(a1_d, size * sizeof(double),
                           a + idx_1, n * sizeof(double),
                           size * sizeof(double),
                           size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy2D(a2_d, size * sizeof(double),
                           a + idx_2, n * sizeof(double),
                           size * sizeof(double),
                           size, cudaMemcpyHostToDevice));

    // if there are more than 512 threads per block, increase blocks and cap at 512 threads
    dim3 threadsPerBlock(size, size);
    dim3 blocksPerGrid(1, 1);
    if (size > 32) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        blocksPerGrid.x = ceil(double(size) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(size) / double(threadsPerBlock.y));
    }
    // printf("Threads per block: (%d %d), Blocks per grid: (%d %d)\n",
    //        threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y);

    // execute kernel
    multMat_kernel<<<blocksPerGrid, threadsPerBlock>>>(a1_d, a2_d, size, res_d);
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());

    // copy and return calculations
    if (copyToFirst == 0) {
        gpuErrchk(cudaMemcpy2D(a + idx_1, n * sizeof(double),
                               res_d, size * sizeof(double),
                               size * sizeof(double),
                               size, cudaMemcpyDeviceToHost));
    } else {
        gpuErrchk(cudaMemcpy2D(a + idx_2, n * sizeof(double),
                               res_d, size * sizeof(double),
                               size * sizeof(double),
                               size, cudaMemcpyDeviceToHost));
    }

    // printSubMat(a, idx_2, size, n, "Result");
    // free device memory
    cudaFree(a1_d);
    cudaFree(a2_d);
    cudaFree(res_d);
    return;
}

// Recursively invert A11, A22, and then A21
void parallelInverseRecurse(double *a, int idx, int size, int n) {
    // Base Case: Invert if a is a simple 2x2 matrix
    if (size == 2) {
        invertTwoByTwo(a, idx, n);
        return;
    }

    // Calculate starting index for the 3 submatrices and make recursive calls
    int a11_idx = idx;                                // upper left submatrix
    int a22_idx = idx + size / 2 * n + size / 2;      // lower right submatrix
    int a21_idx = idx + size / 2 * n;                 // bottom left full submatrix
    parallelInverseRecurse(a, idx, size / 2, n);      // recurse on upper left submatrix
    parallelInverseRecurse(a, a22_idx, size / 2, n);  // recurse on bottom right submatrix

    // Invert A21
    parallelMultMat(a, a22_idx, a21_idx, size / 2, n, 1);  // A22 * A21 & put result into A21
    parallelMultMat(a, a21_idx, a11_idx, size / 2, n, 0);  // A21 * A11 & put result into A21
    addNegative(a, a21_idx, size / 2, n);                  // Add negative sign to A21
}

int cpu_recursive_gpu_mult(int inputSize, int check, int debug) {
    printf("\nCPU Recursive, GPU Multiplication\n");
    // Initialize
    float gpu_time;
    int n = pow(2, inputSize);  // Matrix size: 2^x = n
    double *a, *a_old, *b, *b_old;
    a = initMat(n);
    a_old = initMat(n);
    copyMat(a_old, a, 0, n, n);
    b = initVec(n);
    b_old = initVec(n);
    copyVec(b_old, b, n);

    // Begin Inversion
    gstart();
    parallelInverseRecurse(a, 0, n, n);
    gend(&gpu_time);
    printf("Total Time: %f\n", gpu_time);
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
