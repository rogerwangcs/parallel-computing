#include <stdio.h>
#define N 3
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

// failed to figure this out
__host__ __device__ void printMat2(int mat[M][N]) {
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
    // pB[blockIdx.x][threadIdx.x] = pA[threadIdx.x][blockIdx.x];  // <-- this caused the weird zeros
    pB[threadIdx.x][blockIdx.x] = pA[blockIdx.x][threadIdx.x];  // <-- this is your fix
}

int main() {
    // Allocate memory for array on host
    int A[N][M] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    int B[M][N];

    printf("Matrix:\n");
    printMat(A);

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
    printMat2(B);

    return 0;
}