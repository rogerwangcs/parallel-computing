#include <stdio.h>
#include "../timerc.h"

__global__ void cumulative_sum(int *in, int *partcum, int size) {
    int *shifted_in = in + blockIdx.x * 2 * blockDim.x;
    int local_ix = threadIdx.x;

    for (int s = 1; s <= blockDim.x; s *= 2) {
        if (local_ix < blockDim.x / s) {
            int temp = local_ix * 2 * s;
            shifted_in[temp + s - 1 + s] = shifted_in[temp + s - 1] + shifted_in[temp + s - 1 + s];
        }
        __syncthreads();
    }

    for (int s = blockDim.x / 2; s >= 1; s = s / 2) {
        if (local_ix < blockDim.x / s - 1) {
            int temp = local_ix * 2 * s;
            shifted_in[2 * s - 1 + s + temp] = shifted_in[2 * s - 1 + temp] + shifted_in[2 * s - 1 + s + temp];
        }

        __syncthreads();
    }

    if (local_ix == 0 && partcum != NULL) {
        partcum[blockIdx.x] = shifted_in[2 * blockDim.x - 1];
    }
}

__global__ void fixcumsum(int *block_cum_sum, int *small_part_cum, int size_small_part, int size) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;

    if (ix >= size_small_part) {
        block_cum_sum[threadIdx.x] = block_cum_sum[threadIdx.x] + small_part_cum[-1 + ix / size_small_part];
    }
}

int main() {
    int n = 64 * 1024 * 1024;
    int num_threads_per_block = 1024;
    int num_blocks = n / (2 * num_threads_per_block);
    int *h_v = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_v[i] = 1;
    }

    int *h_c_v = (int *)malloc(n * sizeof(int));
    float cpu_time;
    cstart();
    h_c_v[0] = h_v[0];
    for (int i = 1; i < n; i++) {
        h_c_v[i] = h_c_v[i - 1] + h_v[i];
    }
    cend(&cpu_time);
    printf("CPU time cum sum = %f\n", cpu_time);

    int *d_v;
    int *d_part_cum;
    int *d_part_part_cum;
    cudaMalloc((void **)&d_v, (n * sizeof(int)));
    cudaMalloc((void **)&d_part_cum, (num_blocks * sizeof(int)));
    cudaMalloc((void **)&d_part_part_cum, (num_blocks / (2 * num_threads_per_block) * sizeof(int)));

    cudaMemcpy(d_v, h_v, (n * sizeof(int)), cudaMemcpyHostToDevice);

    float gpu_time;
    gstart();
    cumulative_sum<<<num_blocks, num_threads_per_block>>>(d_v, d_part_cum, n);
    gend(&gpu_time);
    printf("GPU time cum sum = %f\n", gpu_time);

    cudaMemcpy(h_v, d_v, (n * sizeof(int)), cudaMemcpyDeviceToHost);

    printf("\n-----\n");
    for (int i = 0; i < 4 * num_threads_per_block; i++) {
        printf(" %d ", h_v[i]);
    }

    int *h_part_cum = (int *)malloc(num_blocks * sizeof(int));
    cudaMemcpy(h_part_cum, d_part_cum, (num_blocks * sizeof(int)), cudaMemcpyDeviceToHost);

    printf("\n-----\n");
    for (int i = 0; i < num_blocks; i++) {
        printf(" %d ", h_part_cum[i]);
    }

    cumulative_sum<<<num_blocks / (2 * num_threads_per_block), num_threads_per_block>>>(d_part_cum, d_part_part_cum, num_blocks);

    int *h_part_part_cum = (int *)malloc((num_blocks / (2 * num_threads_per_block)) * sizeof(int));
    cudaMemcpy(h_part_part_cum, d_part_part_cum, ((num_blocks / (2 * num_threads_per_block)) * sizeof(int)), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_part_cum, d_part_cum, (num_blocks * sizeof(int)), cudaMemcpyDeviceToHost);

    printf("\n-----\n");

    for (int i = 0; i < 4 * num_threads_per_block; i++) {
        printf(" %d ", h_part_cum[i]);
    }

    printf("\n-----\n");
    for (int i = 0; i < (num_blocks / (2 * num_threads_per_block)); i++) {
        printf(" %d ", h_part_part_cum[i]);
    }

    cumulative_sum<<<1, (num_blocks / (2 * num_threads_per_block))>>>(d_part_part_cum, NULL, num_blocks / (2 * num_threads_per_block));

    fixcumsum<<<(num_blocks / (num_threads_per_block)), num_threads_per_block>>>(d_part_cum, d_part_part_cum, 2 * num_threads_per_block, num_blocks);

    fixcumsum<<<(n / (num_threads_per_block)), num_threads_per_block>>>(d_v, d_part_cum, 2 * num_threads_per_block, n);

    return 0;
}
