#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../timerc.h"

/*--------------------------------------------------------------
 *    Matrix Multiplication GPU with dim3 and blocks

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

// Helper Functions
void __host__ __device__ printSubMat(double *a, int idx, int size, int n, const char *title) {
    printf("%s\n", title);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int local_idx = idx + i * n + j;
            printf("%5.2f ", a[local_idx]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMat(double *a, int n, const char *title) {
    printf("%s\n", title);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%5.2f ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

double *initMat(int n) {
    double *a = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i >= j)
                a[i * n + j] = i + j;
            else
                a[i * n + j] = 0;
        }
    }
    return a;
}

void copyMat(double *newA, double *a, int idx, int size, int n) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int local_idx = idx + i * n + j;
            a[local_idx] = newA[i * size + j];
        }
    }
}

void multMat(double *a, int idx_1, int idx_2, int size, int n, double *res) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            res[i * size + j] = 0;
            for (int k = 0; k < size; k++)
                res[i * size + j] += a[idx_1 + i * n + k] * a[idx_2 + k * n + j];
        }
    }
}

__global__ void multMat_kernel(double *a, int idx_1, int idx_2, int size, int n, double *res) {
    int i = gridDim.x * blockIdx.y + blockIdx.x;
    int j = blockDim.x * threadIdx.y + threadIdx.x;
    res[i * size + j] = 0.0;
    for (int k = 0; k < size; k++) {
        res[i * size + j] += a[idx_1 + i * n + k] * a[idx_2 + k * n + j];
    }
    return;
}
__global__ void multMat_kernel(double *a1, double *a2, int size, double *res) {
    int i = gridDim.x * blockIdx.y + blockIdx.x;
    int j = blockDim.x * threadIdx.y + threadIdx.x;
    res[i * size + j] = 0.0;
    for (int k = 0; k < size; k++) {
        res[i * size + j] += a1[i * size + k] * a2[k * size + j];
    }
    return;
}

void parallelMultMat(double *a, int idx_1, int idx_2, int size, int n, double *res, int copyToFirst) {
    // device variables
    double *a1_d, *a2_d, *res_d;
    gpuErrchk(cudaMalloc((void **)&a1_d, size * size * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&a2_d, size * size * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&res_d, size * size * sizeof(double)));
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
    printf("Threads per block: (%d %d), Blocks per grid: (%d %d)\n",
           threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y);

    // execute kernel
    multMat_kernel<<<blocksPerGrid, threadsPerBlock>>>(a1_d, a2_d, size, res_d);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

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

int main() {
    int n = pow(2, 11);  // Matrix size: 2^x = n

    // host variables
    double *a, *a2, *res;
    a = initMat(n);
    a2 = initMat(n);
    res = (double *)malloc(n * n * sizeof(double));

    // printMat(a, n, "A:");
    // multMatParallel(a, 0, 0, n / 2, n, res);
    // printMat(res, n / 2, "Res:");

    // printMat(a, n, "A:");
    parallelMultMat(a, 0, 0, n, n, res, 0);
    // printMat(res, n, "Res:");

    free(a);
    free(a2);
    free(res);
    return 0;
}
