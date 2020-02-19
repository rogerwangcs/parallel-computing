#include <stdio.h>

texture<int> tex;
texture<int, 2> tex2D;

__host__ __device__ void printArr(int *arr, int n, const char *name) {
    int max = 10;
    if (n < max) {
        max = n;
    }
    printf("[%s]: ", name);
    for (int i = 0; i < max; i++) {
        printf("%d ", arr[i]);
    }
    printf("... \n");
}

__global__ void texread() {
    int s = tex1Dfetch(tex, 1);
    printf("%d  \n", s);
}

__global__ void texread2D() {
    int s = tex2D<int>(tex2D, 0, 0);  // (x, y)
    printf("%d  \n", s);
}

int main() {
    int p = 8;
    int n = p * p;
    int *d_v;

    cudaMalloc((void **)&d_v, n * sizeof(int));
    int *h_v = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        h_v[i] = 1;
    }
    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    cudaBindTexture(NULL, tex, d_v, n * sizeof(int));
    cudaBindTexture2D(NULL, tex2D, d_v, desc, p, p, p * sizeof(int));

    texread<<<1, 1>>>();
    cudaMemcpy(h_v, d_v, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free memory
    cudaUnbindTexture(tex);
    free(h_v);
    cudaFree(d_v);

    return 0;
}