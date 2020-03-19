#include <stdio.h>
#include "../timerc.h"

__global__ void sum1(int *a, int *b, int n, int c) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  // 1D block, 1D grid
    int local_total = 0;
    if (c * ix < n) {
        int limit = c * ix + (c - 1);
        if (limit >= n) {
            limit = n - 1;
        }
        for (int i = c * ix; i <= limit; i++) {
            local_total += a[i];
        }
        b[ix] = local_total;
    }
}

__global__ void sum2(int *in, int *out, int size) {
    int *shifted_in = in + blockIdx.x * 2 * blockDim.x;
    int local_ix = threadIdx.x;

    for (int s = 1; s <= blockDim.x; s = s * 2) {
        if (local_ix < blockDim.x / s) {
            shifted_in[2 * s * local_ix] = shifted_in[2 * s * local_ix] + shifted_in[2 * s * local_ix + s];
        }

        __syncthreads();
    }

    if (local_ix == 0) {
        out[blockIdx.x] = shifted_in[0];
    }
}

int main() {
    float time_it, time_it2, time_it3;
    int size = 64 * 1024 * 1024;
    int blocksize = 1024;
    int gridsize = size / (2 * blocksize);
    int c = 64;

    int *v = (int *)malloc(size * sizeof(int));
    int *h_d_v;
    int *h_d_o;
    cudaMalloc((void **)&h_d_v, size * sizeof(int));
    cudaMalloc((void **)&h_d_o, (gridsize) * sizeof(int));

    for (int i = 0; i < size; i++) {
        v[i] = 1;
    }
    int total = 0;
    cstart();
    for (int i = 0; i < size; i++) {
        total += v[i];
    }
    cend(&time_it);
    printf("CPU:   %f        %d\n", time_it, total);

    printf("====================================\n");
    printf("      C         TIME        TOTAL\n");
    printf("====================================\n");
    cudaMemcpy(h_d_v, v, size * sizeof(int), cudaMemcpyHostToDevice);

    int cValues[7] = {1, 16, 32, 64, size / 1024, size / 64, size};
    for (int i = 0; i < 7; i++) {
        int c = cValues[i];
        gstart();
        sum1<<<1024, 1024>>>(h_d_v, h_d_o, size, c);
        cudaMemcpy(v, h_d_o, (gridsize) * sizeof(int), cudaMemcpyDeviceToHost);
        gend(&time_it);

        cstart();
        int gpu_total = 0;
        for (int i = 0; i < size; i++) {
            gpu_total += v[i];
        }
        cend(&time_it2);

        printf("%8d   %f     %d\n", c, time_it + time_it2, gpu_total);
    }

    printf("===========================================\n");
    printf(" GRIDSIZE   BLOCKSIZE       TIME    TOTAL\n");
    printf("===========================================\n");

    int blockSizes[2] = {1024, 64};
    int gridSizes[2] = {size / (2 * blocksize), size / (4 * blocksize)};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int blockSize = blockSizes[i];
            int gridSize = gridSizes[j];

            gstart();
            sum2<<<gridSize, blockSize>>>(h_d_v, h_d_o, size);
            cudaMemcpy(v, h_d_o, (gridsize) * sizeof(int), cudaMemcpyDeviceToHost);
            gend(&time_it);

            cstart();
            int gpu_total = 0;
            for (int i = 0; i < size; i++) {
                gpu_total += v[i];
            }
            cend(&time_it2);

            printf("%d       %d       %f       %d\n", gridSize, blockSize, time_it, gpu_total);
        }
    }

    free(v);
    cudaFree(h_d_v);

    return 0;
}
