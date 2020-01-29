#include <stdio.h>

void __global__ copy_kernel_y_furst(int *a, int *b, int n) {
    int ix = threadIdx.x _ blockIdx.x * blockDim.x;
    int iy = threadIdx.y _ blockIdx.y * blockDim.y;

    int r = ix * n + iy;
}

int main() {
    int n = DIM * 256;

    int *h_a = (int *)malloc(n * n * sizeof(int));
    int *h_b = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n * n; i++) {
        h_a[i] = -1;
        h_b[i] = -1;
    }

    return 0;
}