#include <stdio.h>
#define N 1
#define M 4
__host__ __device__ void printMat(int mat[N][M]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

__host__ __device__ void printMatT(int mat[M][N]) {
    int i, j;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void matTranspose(int (*pA)[M], int (*pB)[N]) {
    printf("block id: %d, thread id: %d\n", blockIdx.x, threadIdx.x);
    pB[blockIdx.x][threadIdx.x] = pA[threadIdx.x][blockIdx.x];
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printMatT(pB);
}

int main() {
    // Allocate memory for array on host
    // int A[N][M] = {{1, 2, 3}, {5, 6, 7}, {9, 10, 11}};
    int A[N][M] = {{1, 2, 3, 4}};
    int B[M][N] = {{100}, {100}, {100}, {100}};

    printf("Matrix:\n");
    printMat(A);
    printMatT(B);

    int(*pA)[M], (*pB)[N];

    // Allocate memory for array on device
    cudaMalloc((void **)&pA, (N * M) * sizeof(int));
    cudaMalloc((void **)&pB, (M * N) * sizeof(int));

    // Copy data from host array to device array
    cudaMemcpy(pA, A, (N * M) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pB, B, (N * M) * sizeof(int), cudaMemcpyHostToDevice);

    matTranspose<<<N, M>>>(pA, pB);
    cudaDeviceSynchronize();

    // Copy data from device array to host array
    cudaMemcpy(B, pB, (N * M) * sizeof(int), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(pA);
    cudaFree(pB);

    printf("Transposed:\n");
    printMatT(B);

    return 0;
}