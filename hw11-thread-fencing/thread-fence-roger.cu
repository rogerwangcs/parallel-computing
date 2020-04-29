// Roger Wang
#include <stdio.h>

#include "../timerc.h"

__global__ void reduce_with_atomics(int* input, int* output) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = input[i] + input[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__device__ unsigned int timer_counter = 0;

__global__ void reduce_with_atomics_and_fences(int* input, volatile int* output) {
    __shared__ bool amILast;  // one variable per block. Every thread in the block sees this variable

    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;

    sdata[tid] = input[i] + input[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
    __threadfence();
    if (tid == 0) {
        unsigned int ticket = atomicInc(&timer_counter, gridDim.x);
        amILast = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (amILast) {
        if (tid < gridDim.x / 2) {
            sdata[tid] = output[tid] + output[tid + gridDim.x / 2];
        }
        __syncthreads();

        for (int s = gridDim.x / 4; s > 0; s = s / 2) {
            if (tid < s) {
                sdata[tid] = sdata[tid] + sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[0] = sdata[0];
            timer_counter = 0;
        }
    }
}

__device__ int d_o = 0;

int main() {
    int n = 1024 * 1024 * 4;  // 67108864
    int num_blocks = n / (2 * 1024);
    int* h_d = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        h_d[i] = 1;
    }

    float cpu_time;
    cstart();
    int cpu_total = 0;
    for (int i = 0; i < n; i++) {
        cpu_total += h_d[i];
    }
    cend(&cpu_time);
    printf("Value %d in %f time on the CPU\n", cpu_total, cpu_time);

    int* d_d;
    cudaMalloc((void**)&d_d, n * sizeof(int));
    cudaMemcpy(d_d, h_d, n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_o_ptr;
    cudaGetSymbolAddress((void**)&d_o_ptr, d_o);

    float gpu_time;
    gstart();
    reduce_with_atomics<<<num_blocks, 1024, 2048 * sizeof(int)>>>(d_d, d_o_ptr);
    gend(&gpu_time);

    int h_o;
    cudaMemcpy(&h_o, d_o_ptr, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Value %d in %f time using atomics\n", h_o, gpu_time);

    cudaDeviceSynchronize();

    cudaFree(d_o_ptr);
    cudaFree(d_d);
    free(h_d);
    return 0;
}
