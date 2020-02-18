#include <stdio.h>
#include "../timerc.h"

__host__ __device__ void printArr(int *arr, int n) {
    if (n > 10) {
        n = 10;
    }
    printf("v[0:10]:   ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

__global__ void partialSum1(int *a, int *b, int n, int c) {
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

int main() {
    printf("Author: Roger Wang\n");
    printf("==================\n");

    int size = 1024 * 1024;
    int c = 1024;  // c must be at least blocksize
    int blocksize = 32;
    int gridsize = size / (blocksize);
    int sizeReduced = ceil(size / c);

    int *v = (int *)malloc(size * sizeof(int));
    int *o = (int *)malloc(size * sizeof(int));
    int *h_d_v;
    int *h_d_o;
    cudaMalloc((void **)&h_d_v, size * sizeof(int));
    cudaMalloc((void **)&h_d_o, (gridsize) * sizeof(int));

    // initialize
    for (int i = 0; i < size; i++) {
        v[i] = 1;
    }

    float cpu_timer, gpu_timer1, gpu_timer2;
    int total = 0;

    // Single pass parallel sum
    printf("\n\nSingle Pass Parallel Sum\n");
    printf("========================\n");

    cudaMemcpy(h_d_v, v, size * sizeof(int), cudaMemcpyHostToDevice);
    gstart();
    partialSum1<<<gridsize, blocksize>>>(h_d_v, h_d_o, size, c);
    gend(&gpu_timer1);
    cudaMemcpy(o, h_d_o, (gridsize) * sizeof(int), cudaMemcpyDeviceToHost);
    printArr(o, size);

    cstart();
    total = 0;
    for (int i = 0; i < gridsize; i++) {
        total += o[i];
    }
    cend(&cpu_timer);

    printf("Correct Total: %d\n", size);
    printf("Calculated Total: %d\n", total);
    printf("Time: %f\n", cpu_timer + gpu_timer1);
    // Single pass parallel sum

    // Two pass parallel sum
    printf("\n\nTwo Pass Parallel Sum\n");
    printf("======================\n");

    cudaMemcpy(h_d_v, v, size * sizeof(int), cudaMemcpyHostToDevice);
    gstart();
    partialSum1<<<gridsize, blocksize>>>(h_d_v, h_d_o, size, c);
    gend(&gpu_timer1);
    cudaMemcpy(o, h_d_o, (gridsize) * sizeof(int), cudaMemcpyDeviceToHost);

    printArr(o, size);

    cudaMemcpy(h_d_v, o, size * sizeof(int), cudaMemcpyHostToDevice);
    gstart();
    partialSum1<<<gridsize, blocksize>>>(h_d_v, h_d_o, gridsize, c);
    gend(&gpu_timer2);
    cudaMemcpy(o, h_d_o, (gridsize) * sizeof(int), cudaMemcpyDeviceToHost);

    printArr(o, size);

    cstart();
    total = 0;
    for (int i = 0; i < gridsize / c; i++) {
        total += o[i];
    }
    cend(&cpu_timer);

    printf("Correct Total: %d\n", size);
    printf("Calculated Total: %d\n", total);
    printf("Time: %f\n", cpu_timer + gpu_timer1 + gpu_timer2);
    // Two pass parallel sum

    free(v);
    cudaFree(h_d_v);

    return 0;
}
