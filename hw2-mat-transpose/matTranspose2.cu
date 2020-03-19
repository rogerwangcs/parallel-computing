#include <stdio.h>

__host__ __device__ void printFlatMat(int *mat, int n, int m) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < m; col++) {
            printf("%d ", mat[m * row + col]);
        }
        printf("\n");
    }
    printf("\n");
}

__host__ __device__ void printFlatMatColOrder(int *mat, int n, int m) {
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            printf("%d ", mat[n * row + col]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void matTranspose(int n, int m, int *dOut, int *dIn) {
    // printf("block id: %d, thread id: %d\n", blockIdx.x, threadIdx.x);
    dOut[n * threadIdx.x + blockIdx.x] = dIn[m * blockIdx.x + threadIdx.x];
}

int main() {
    int N = 3;
    int M = 4;
    int A[N * M] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int B[N * M];

    printFlatMat(A, N, M);

    // Declare GPU pointers
    int *hA, *hB;

    // Allocate memory for array on device
    cudaMalloc((void **)&hA, (N * M) * sizeof(int));
    cudaMalloc((void **)&hB, (N * M) * sizeof(int));

    // Copy data from host array to device array
    cudaMemcpy(hA, A, (N * M) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hB, B, (N * M) * sizeof(int), cudaMemcpyHostToDevice);

    matTranspose<<<N, M>>>(N, M, hB, hA);
    cudaDeviceSynchronize();

    // Copy data from device array to host array
    cudaMemcpy(B, hB, (N * M) * sizeof(int), cudaMemcpyDeviceToHost);

    printFlatMat(B, M, N);

    // Free GPU pointers
    cudaFree(hA);
    cudaFree(hB);

    return 0;
}