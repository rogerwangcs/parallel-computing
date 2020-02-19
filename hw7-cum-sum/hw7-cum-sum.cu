// Author: Roger Wang

#include <stdio.h>
#include "../timerc.h"

void initArr(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = 1;
    }
}

__host__ __device__ void arrHeadTail(int *arr, int n, const char *name) {
    int max = 10;
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

__global__ void cumulative_sum(int *in, int *pc1, int size) {
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

    if (local_ix == 0 && pc1 != NULL) {
        pc1[blockIdx.x] = local_block_ix[2 * blockDim.x - 1];
    }
}

__global__ void fixcumsum(int *pc1, int *pc2, int pc2_size, int size) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;

    if (ix >= pc2_size) {
        pc1[ix] = pc1[ix] + pc2[-1 + ix / pc2_size];
    }
}

int main() {
    printf("Author: Roger Wang\n");
    printf("==================\n");

    // Initialize
    int n = 64 * 1024 * 1024;
    int num_threads = 1024;
    int num_blocks = n / (2 * num_threads);

    float cpu_time, gpu_time1, gpu_time2;

    // Initialize array
    int *h_v = (int *)malloc(n * sizeof(int));
    initArr(h_v, n);
    arrHeadTail(h_v, n, "h_v");

    // CPU Cum Sum
    int *h_c_v = (int *)malloc(n * sizeof(int));
    cstart();
    h_c_v[0] = h_v[0];
    for (int i = 1; i < n; i++) {
        h_c_v[i] = h_c_v[i - 1] + h_v[i];
    }
    arrHeadTail(h_c_v, n, "h_c_v");
    cend(&cpu_time);
    printf("CPU cum sum time: %f\n", cpu_time);
    printf("========================= \n");
    // CPU Cum Sum

    // GPU Cum Sum ----------------------------------------------
    initArr(h_v, n);
    arrHeadTail(h_v, n, "h_v");

    gstart();
    // allocate cuda memory
    int *d_v, *d_pc1, *d_pc2;
    cudaMalloc((void **)&d_v, n * sizeof(int));
    cudaMalloc((void **)&d_pc1, (num_blocks * sizeof(int)));
    cudaMalloc((void **)&d_pc2, (num_blocks / (2 * num_threads) * sizeof(int)));

    // run first pass of cum sum to get partial sum 1 (pc1)
    cudaMemcpy(d_v, h_v, n * sizeof(int), cudaMemcpyHostToDevice);
    cumulative_sum<<<num_blocks, num_threads>>>(d_v, d_pc1, n);
    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);
    arrHeadTail(h_v, 4 * num_threads, "h_v");

    // get array of last element of each partial sum block
    int *h_pc1 = (int *)malloc(num_blocks * sizeof(int));
    cudaMemcpy(h_pc1, d_pc1, (num_blocks * sizeof(int)), cudaMemcpyDeviceToHost);
    arrHeadTail(h_pc1, num_blocks, "h_pc1");

    // run second pass of cum sum to sum to pc2
    cumulative_sum<<<num_blocks / (2 * num_threads), num_threads>>>(d_pc1, d_pc2, num_blocks);
    int *h_pc2 = (int *)malloc((num_blocks / (2 * num_threads)) * sizeof(int));
    cudaMemcpy(h_pc2, d_pc2, ((num_blocks / (2 * num_threads)) * sizeof(int)), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pc1, d_pc1, (num_blocks * sizeof(int)), cudaMemcpyDeviceToHost);
    arrHeadTail(h_pc2, num_blocks / (2 * num_threads), "h_pc2");
    arrHeadTail(h_pc1, num_blocks, "h_pc1");

    cumulative_sum<<<1, (num_blocks / (2 * num_threads))>>>(d_pc2, NULL, num_blocks / (2 * num_threads));

    fixcumsum<<<(num_blocks / (num_threads)), num_threads>>>(d_pc1, d_pc2, 2 * num_threads, num_blocks);
    fixcumsum<<<(n / (num_threads)), num_threads>>>(d_v, d_pc1, 2 * num_threads, n);

    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);

    gend(&gpu_time1);
    arrHeadTail(h_v, n, "h_v");
    printf("GPU cum sum time (with memcpy): %f\n", gpu_time1);
    printf("========================= \n");
    // GPU Cum Sum ----------------------------------------------

    cudaFree(d_v);
    cudaFree(d_pc1);
    cudaFree(d_pc2);
    free(h_v);
    free(h_c_v);
    free(h_pc1);
    free(h_pc2);

    return 0;
}
