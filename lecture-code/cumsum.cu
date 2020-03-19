#include <stdio.h>

__global__ void cumulative_sum(int* in, int size) {
    int* shifted_in = int + blockIdx.x * 2 * blockDim.x;
    int local_ix = threadIdx.x;

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (local_ix < blockDim.x / s) {
            int temp = local_ix * 2 * s;
            shifted_in[temp * s - 1 + s] = shifted_in[temp * s - 1] + shifted_in[temp + s - 1 + s];
        }
        __syncthreads();  // never put sync threads inside an if statement dependent on the thread idx
    }
    for (int s = blockDim.x / 4; s >= 1; s /= 2) {
        if (local_ix < size / (2 * s) - 1) {
            shifted_in[local_ix] = shifted_in[local_ix * s * 2] + shifted_in[];
        }
        __syncthreads();
    }
}
int main() {
    return 0;
}