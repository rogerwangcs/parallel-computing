// Allocate memory for array on host
// Allocate memory for array on device
// Fill array on host
// Copy data from host array to device array
// Do something on device (e.g. vector addition)
// Copy data from device array to host array
// Check data for correctness
// Free Host Memory
// Free Device Memory

#include <stdio.h>
#define N 2

__host__ __device__ void printMat(int mat[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
    // printf("\n");
}

__global__ void matAdd(int pA[N][N], int pB[N][N], int pC[N][N]) {
    pC[threadIdx.x][threadIdx.y] = pA[threadIdx.x][threadIdx.y] + pB[threadIdx.x][threadIdx.y];
}

int main() {
    // Allocate memory for array on host
    int A[N][N] = {{1, 1}, {0, 0}};
    int B[N][N] = {{0, 0}, {2, 2}};
    int C[N][N];

    printMat(A);
    printMat(B);

    int(*pA)[N], (*pB)[N], (*pC)[N];

    // Allocate memory for array on device
    cudaMalloc((void**)&pA, (N * N) * sizeof(int));
    cudaMalloc((void**)&pB, (N * N) * sizeof(int));
    cudaMalloc((void**)&pC, (N * N) * sizeof(int));

    // Copy data from host array to device array
    cudaMemcpy(pA, A, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pB, B, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pC, C, (N * N) * sizeof(int), cudaMemcpyHostToDevice);

    matAdd<<<1, {N, N}>>>(pA, pB, pC);
    cudaDeviceSynchronize();

    // Copy data from device array to host array
    cudaMemcpy(C, pC, (N * N) * sizeof(int), cudaMemcpyDeviceToHost);

    printMat(C);

    cudaFree(pA);
    cudaFree(pB);
    return 0;
}