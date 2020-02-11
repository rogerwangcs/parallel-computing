// Author: Roger Wang

// to run with L1 cache on:
// nvcc --run file.cu
// to run with L1 cache disabled:
// nvcc --run -Xptxas -dlcm=ca file.cu

#include <stdio.h>
#include "../timerc.h"

#define MAX_DIM 64

void __global__ transpose_kernel_read_and_write_coalesced_with_shared_memory_less_conflicts(int *a, int *b, int n, unsigned int DIM, unsigned int REP, unsigned int OFFSET) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int ix2 = threadIdx.x + blockIdx.y * blockDim.y;
    int iy2 = threadIdx.y + blockIdx.x * blockDim.x;

    int r = iy * n + ix;
    int s = iy2 * n + ix2;

    __shared__ int tmpmat[(MAX_DIM) * (MAX_DIM)];  // 1D shared memory
    tmpmat[threadIdx.y + threadIdx.x * (DIM + OFFSET)] = b[r];

    __syncthreads();  // forces all threads inside the block to reach here. All memory transactions are visible as completed to other threads
        // no synchronism between threads in different blocks.

    a[s] = tmpmat[threadIdx.x + threadIdx.y * (DIM + OFFSET)];
}

int main() {
    float cpu_time;
    float gpu_time;
    int n = 32 * 256;

    int *h_a = (int *)malloc(n * n * sizeof(int));
    int *h_b = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n * n; i++) {
        h_b[i] = i;
        h_a[i] = -1;
    }

    // Testing cpu transpose time
    cstart();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_a[j + n * i] = h_b[i + n * j];
        }
    }
    cend(&cpu_time);
    printf("CPU transpose, DIM=32 : %f\n", cpu_time);
    int OFFSETS[5] = {0, 1, 16, 32, 64};

    printf("BLOCK OFFSET  Transpose Time\n");
    printf("============================\n");

    for (int offset_idx = 0; offset_idx < 5; offset_idx++) {
        // for (int rep_idx = 0; rep_idx < 4; rep_idx++) {
        int *h_a = (int *)malloc(n * n * sizeof(int));
        int *h_b = (int *)malloc(n * n * sizeof(int));
        int *h_d_a, *h_d_b;
        cudaMalloc((void **)&h_d_a, n * n * sizeof(int));
        cudaMalloc((void **)&h_d_b, n * n * sizeof(int));
        cudaMemcpy(h_d_b, h_b, n * n * sizeof(int), cudaMemcpyHostToDevice);

        unsigned int OFFSET = OFFSETS[offset_idx];
        unsigned int DIM = 32;
        unsigned int REP = 4;

        printf("     %d     ", OFFSET);

        dim3 threadsPerBlock(DIM, DIM / REP);
        dim3 blocksPerGrid(n / DIM, n / DIM);

        gstart();
        transpose_kernel_read_and_write_coalesced_with_shared_memory_less_conflicts<<<blocksPerGrid, {DIM, DIM}>>>(h_d_a, h_d_b, n, DIM, REP, OFFSET);
        gend(&gpu_time);
        printf("     %f\n", gpu_time);

        free(h_a);
        free(h_b);
        cudaFree(h_d_a);
        cudaFree(h_d_b);
        // }
    }
    // cudaDeviceSynchronize();

    return 0;
}
