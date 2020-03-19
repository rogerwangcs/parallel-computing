#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../timerc.h"

/*--------------------------------------------------------------
 *    Matrix Multiplication GPU

 *    Author: Roger Wang
 *-------------------------------------------------------------- */

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
                a[i * n + j] = i * n + j + 1;
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
    int i = blockIdx.x;
    int j = threadIdx.x;
    res[i * size + j] = 0.0;
    for (int k = 0; k < size; k++) {
        res[i * size + j] += a[idx_1 + i * n + k] * a[idx_2 + k * n + j];
    }
    return;
}

void multMatParallel(double *a, int idx_1, int idx_2, int size, int n, double *res) {
    // device variables
    double *a_d;
    double *res_d;
    cudaMalloc((void **)&a_d, n * n * sizeof(double));
    cudaMalloc((void **)&res_d, n * n * sizeof(double));
    cudaMemcpy(a_d, a, n * n * sizeof(double), cudaMemcpyHostToDevice);

    // execute kernel
    multMat_kernel<<<size, size>>>(a_d, idx_1, idx_2, size, n, res_d);

    // copy and return calculations
    cudaMemcpy(res, res_d, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free device memory
    cudaFree(a_d);
    cudaFree(res_d);
    return;
}

int main() {
    int n = pow(2, 3);  // Matrix size: 2^x = n

    // host variables
    double *a, *a2, *res;
    a = initMat(n);
    a2 = initMat(n);
    res = (double *)malloc(n * n * sizeof(double));

    printMat(a, n, "A:");
    multMatParallel(a, 0, n / 2, n / 2, n, res);
    printMat(res, n / 2, "Res:");

    free(a);
    free(a2);
    free(res);
    return 0;
}
