#include <stdio.h>

__global__ void cumulative_sum(int* in, int* partcum, int size) {
    int* shifted_in = in + blockIdx.x * 2 * blockDim.x;
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

    if (local_ix == 0) {
        partcum[blockIdx.x] = shifted_in[2 * blockDim.x - 1];
    }
}

int main() {
    int n = 64 * 1024 * 1024;

    int* h_v = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_v[i] = 1;
    }

    int* h_cv = (int*)malloc(n * sizeof(int));
    h_cv[0] = h_v[0];
    for (int i = 1; i < n; i++) {
        h_cv[i] = h_cv[i - 1] + h_v[i];
    }

    int* d_v;
    cudaMalloc((void**)&d_v, n * sizeof(int));
    cudaMemcpy(d_v, h_v, n * sizeof(int), cudaMemcpyHostToDevice);

    int num_threads_per_block = 1024;
    cumulative_sum<<<n / (2 * num_threads_per_block), num_threads_per_block>>>(d_v, n);
    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4 * num_threads_per_block; i++) {
        printf("%d ", h_v[i]);
    }

    return 0;
}
