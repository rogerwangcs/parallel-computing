#include <stdio.h>
#include <stdlib.h>

#include "helpers.h"

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__host__ __device__ void printSubMat(double *a, int idx, int size, int n, const char *title) {
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

__host__ __device__ void printMat(double *a, int n, const char *title) {
    printf("%s\n", title);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%5.2f ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void printVec(double *b, int n, const char *title) {
    printf("%s\n", title);
    for (int i = 0; i < n; i++) {
        printf("%5.2f ", b[i]);
    }
    printf("\n");
}

double *initVec(int n) {
    double *b = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        b[i] = rand() / (double)RAND_MAX;
    }
    return b;
}

double *initMat(int n) {
    double *a = (double *)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i >= j)
                a[i * n + j] = rand() / (double)RAND_MAX;
            else
                a[i * n + j] = 0;
        }
    }
    return a;
}

void matMult(double *mat1, double *mat1_inv, double *res, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = 0.0;
            for (int k = 0; k < n; k++)
                res[i * n + j] += mat1[i * n + k] * mat1_inv[k * n + j];
        }
    }
}

__host__ __device__ void copyMat(double *newA, double *a, int idx, int size, int n) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int local_idx = idx + i * n + j;
            a[local_idx] = newA[i * size + j];
        }
    }
}

void copyVec(double *newB, double *b, int n) {
    for (int i = 0; i < n; i++) {
        newB[i] = b[i];
    }
}

__host__ __device__ void
multSubMat(double *a_d, int idx_1, int idx_2, int size, int n, double *res_d) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            res_d[i * size + j] = 0;
            for (int k = 0; k < size; k++)
                res_d[i * size + j] += a_d[idx_1 + i * n + k] * a_d[idx_2 + k * n + j];
        }
    }
}

void multMatVec(double *a, double *b, double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 0;
        for (int j = 0; j < n; j++) {
            x[i] += a[i * n + j] * b[j];
        }
    }
}

__host__ __device__ void addNegative(double *a_d, int idx, int size, int n) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int local_idx = idx + i * n + j;
            a_d[local_idx] = -1 * a_d[local_idx];
        }
    }
}

// Base Case 2x2 Inplace inversion
__host__ __device__ void invertTwoByTwo(double *a, int idx, int n) {
    int idx0 = idx;
    int idx1 = idx + 1;
    int idx2 = idx + n;
    int idx3 = idx + n + 1;
    double det = a[idx0] * a[idx3] - a[idx2] * a[idx1];
    double temp;
    temp = a[idx3];
    a[idx3] = a[idx0];
    a[idx0] = temp;
    a[idx2] = -a[idx2];
    a[idx1] = -a[idx1];

    a[idx0] *= (1 / det);
    a[idx1] *= (1 / det);
    a[idx2] *= (1 / det);
    a[idx3] *= (1 / det);
}