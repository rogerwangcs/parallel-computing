// Author: Roger Wang

// to run with L1 cache on:
// nvcc --run file.cu
// to run with L1 cache disabled:
// nvcc --run -Xptxas -dlcm=ca file.cu

#include <stdio.h>
#include "../timerc.h"

#define MAX_DIM 64

void __global__ transpose_kernel_write_coalesced_multi_ele(int *a, int *b, int n, unsigned int DIM, unsigned int REP) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = REP * iy * n + ix;
    int s = ix * n + REP * iy;

    for (int j = 0; j < REP; j++) {
        a[r + j * n] = b[s + j];
    }
}

void __global__ transpose_kernel_read_coalesced_multi_ele(int *a, int *b, int n, unsigned int DIM, unsigned int REP) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = REP * iy * n + ix;
    int s = ix * n + iy * REP;

    for (int j = 0; j < REP; j++) {
        a[s + j] = b[r + n * j];
    }
}

void __global__ transpose_kernel_read_coalesced(int *a, int *b, int n, unsigned int DIM, unsigned int REP) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int ix2 = threadIdx.x + blockIdx.y * blockDim.y;
    int iy2 = threadIdx.y + blockIdx.x * blockDim.x;

    int r = iy * n + ix;
    int s = iy2 * n + ix2;

    __shared__ int tmpmat[MAX_DIM * MAX_DIM];  // 1D shared memory
    tmpmat[threadIdx.y + threadIdx.x * DIM] = b[r];

    __syncthreads();

    a[s] = tmpmat[threadIdx.x + threadIdx.y * DIM];
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

    int DIMS[3] = {16, 32, 64};
    int REPS[4] = {1, 2, 16, 32};

    printf("DIM  REP   Read Coalesced   Write Coalesced   Shared Memory\n");
    printf("===========================================================\n");

    for (int dim_idx = 0; dim_idx < 3; dim_idx++) {
        for (int rep_idx = 0; rep_idx < 4; rep_idx++) {
            int *h_a = (int *)malloc(n * n * sizeof(int));
            int *h_b = (int *)malloc(n * n * sizeof(int));
            int *h_d_a, *h_d_b;
            cudaMalloc((void **)&h_d_a, n * n * sizeof(int));
            cudaMalloc((void **)&h_d_b, n * n * sizeof(int));
            cudaMemcpy(h_d_b, h_b, n * n * sizeof(int), cudaMemcpyHostToDevice);

            unsigned int DIM = DIMS[dim_idx];
            unsigned int REP = REPS[rep_idx];
            printf("%d   %d   ", DIM, REP);

            dim3 threadsPerBlock(DIM, DIM / REP);
            dim3 blocksPerGrid(n / DIM, n / DIM);

            gstart();
            transpose_kernel_write_coalesced_multi_ele<<<blocksPerGrid, threadsPerBlock>>>(h_d_a, h_d_b, n, DIM, REP);
            gend(&gpu_time);
            printf("    %f         ", gpu_time);

            gstart();
            transpose_kernel_read_coalesced_multi_ele<<<blocksPerGrid, threadsPerBlock>>>(h_d_a, h_d_b, n, DIM, REP);
            gend(&gpu_time);
            printf("%f           ", gpu_time);

            gstart();
            transpose_kernel_read_coalesced<<<blocksPerGrid, {DIM, DIM}>>>(h_d_a, h_d_b, n, DIM, REP);
            gend(&gpu_time);
            printf("%f\n", gpu_time);

            free(h_a);
            free(h_b);
            cudaFree(h_d_a);
            cudaFree(h_d_b);
        }
    }
    // cudaDeviceSynchronize();

    return 0;
}
