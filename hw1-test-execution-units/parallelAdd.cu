
#include <stdio.h>
#include <stdlib.h>
#define BLOCKS 3
#define THREADS 3

void printIntArr(int *arr, int size);

__global__ void kernel_fn(int *a, int *b) {
    int i = blockIdx.x;
    if (i < BLOCKS) {
        b[i] = 2 * a[i];
    }
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // Create int arrays on the CPU.
    // ('h' stands for "host".)
    int ha[BLOCKS], hb[BLOCKS];
    // Create corresponding int arrays on the GPU.
    int *da, *db;
    cudaMalloc((void **)&da, BLOCKS * sizeof(int));
    cudaMalloc((void **)&db, BLOCKS * sizeof(int));

    // Initialise the input data on the CPU.
    for (int i = 0; i < BLOCKS; ++i) {
        ha[i] = i;
    }

    // Copy input data to array on GPU.
    // dst, src, size
    cudaMemcpy(da, ha, BLOCKS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch GPU
    kernel_fn<<<BLOCKS, THREADS>>>(da, db);
    // Flush std of gpu
    cudaDeviceSynchronize();

    // Copy output array from GPU back to CPU.
    cudaMemcpy(hb, db, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    printIntArr(hb, BLOCKS);

    // Free up the arrays on the GPU.
    cudaFree(da);
    cudaFree(db);
    return 0;
}

void printIntArr(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}