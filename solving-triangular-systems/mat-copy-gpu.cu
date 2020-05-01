#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void run_kernel(double *a_d, double *b_d, int size) {
    printMat(a_d, size, "a_d");
    for (int i = 0; i < size * size; i++) {
        b_d[i] = a_d[i] * 3;
    }
    printMat(b_d, size, "b_d");
}

int main() {
    double *a, *b;
    int n = 4;
    int size = 2;
    int idx_2 = 8;

    a = initMat(n);
    b = initMat(size);
    printMat(a, n, "A");
    printMat(b, size, "B");

    double *a_d, *b_d;
    cudaMalloc((void **)&a_d, size * size * sizeof(double));
    cudaMalloc((void **)&b_d, size * size * sizeof(double));
    cudaMemcpy2D(a_d, size * sizeof(double), a + idx_2, n * sizeof(double), size * sizeof(double), size, cudaMemcpyHostToDevice);

    run_kernel<<<1, 1>>>(a_d, b_d, size);
    cudaDeviceSynchronize();

    cudaMemcpy2D(a + idx_2, n * sizeof(double), b_d, size * sizeof(double), size * sizeof(double), size, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(b, size * sizeof(double), b_d, size * sizeof(double), size * sizeof(double), size, cudaMemcpyDeviceToHost);
    printSubMat(a, idx_2, size, n, "Result A");
    printMat(b, size, "Result B");

    cudaFree(a_d);
    cudaFree(b_d);
    free(a);
    free(b);
    return 0;
}
