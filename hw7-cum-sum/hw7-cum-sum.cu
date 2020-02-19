#include <stdio.h>
#include "../timerc.h"

__host__ __device__ void arrHeadTail(int *arr, int n, char *name) {
    int max = 20;
    if (n < max) {
        max = n;
    }
    printf("[%s]: ", name);
    for (int i = 0; i < max / 2; i++) {
        printf("%d ", arr[i]);
    }
    printf("... ");
    for (int i = 0; i < max / 2; i++) {
        printf("%d ", arr[n - max / 2 + i]);
    }
    printf("\n");
}

__global__ void cumulative_sum(int *in, int *partcum, int size) {
    int *local_block_ix = in + blockIdx.x * 2 * blockDim.x;
    int local_ix = threadIdx.x;

    for (int s = 1; s <= blockDim.x; s *= 2) {
        if (local_ix < blockDim.x / s) {
            int temp = local_ix * 2 * s;
            local_block_ix[temp + s - 1 + s] = local_block_ix[temp + s - 1] + local_block_ix[temp + s - 1 + s];
        }
        __syncthreads();
    }

    for (int s = blockDim.x / 2; s >= 1; s = s / 2) {
        if (local_ix < blockDim.x / s - 1) {
            int temp = local_ix * 2 * s;
            local_block_ix[2 * s - 1 + s + temp] = local_block_ix[2 * s - 1 + temp] + local_block_ix[2 * s - 1 + s + temp];
        }

        __syncthreads();
    }

    if (local_ix == 0 && partcum != NULL) {
        partcum[blockIdx.x] = local_block_ix[2 * blockDim.x - 1];
    }
}

__global__ void fixcumsum(int *block_cum_sum, int *small_part_cum, int size_small_part, int size) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;

    if (ix >= size_small_part) {
        block_cum_sum[threadIdx.x] = block_cum_sum[threadIdx.x] + small_part_cum[-1 + ix / size_small_part];
    }
}

int main() {
    printf("Author: Roger Wang\n");
    printf("==================\n");

    // Initialize
    int n = 4 * 6;  // 24
    int num_threads = 6;
    int num_blocks = n / (2 * num_threads);  // 2
    // int num_blocks = n / (2 * num_threads);

    int *h_v = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_v[i] = 1;
    }

    // CPU Cum Sum
    int *h_c_v = (int *)malloc(n * sizeof(int));
    float cpu_time;
    cstart();
    h_c_v[0] = h_v[0];
    for (int i = 1; i < n; i++) {
        h_c_v[i] = h_c_v[i - 1] + h_v[i];
    }
    cend(&cpu_time);
    printf("CPU cum sum: %f\n", cpu_time);
    printf("========================= \n");
    // CPU Cum Sum

    // GPU Cum Sum ----------------------------------------------
    int *d_v, *d_pc1, *d_pc2;
    cudaMalloc((void **)&d_v, n * sizeof(int));
    cudaMalloc((void **)&d_pc1, (num_blocks * sizeof(int)));
    cudaMalloc((void **)&d_pc2, (num_blocks / (2 * num_threads) * sizeof(int)));

    cudaMemcpy(d_v, h_v, n * sizeof(int), cudaMemcpyHostToDevice);
    cumulative_sum<<<num_blocks, num_threads>>>(d_v, d_pc1, n);
    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);

    arrHeadTail(h_v, 4 * num_threads, "h_pc1");

    // int *h_pc = (int *)malloc(num_blocks * sizeof(int));
    // cudaMemcpy(h_pc, d_pc1, (num_blocks * sizeof(int)), cudaMemcpyDeviceToHost);

    cumulative_sum<<<1, (num_blocks / (2 * num_threads))>>>(d_pc2, NULL, num_blocks / (2 * num_threads));

    fixcumsum<<<(num_blocks / (num_threads)), num_threads>>>(d_pc1, d_pc2, 2 * num_threads, num_blocks);
    fixcumsum<<<(n / (num_threads)), num_threads>>>(d_v, d_pc1, 2 * num_threads, n);

    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);
    arrHeadTail(h_v, 4 * num_threads, "h_pc1");
    // GPU Cum Sum ----------------------------------------------

    free(h_c_v);

    return 0;
}
