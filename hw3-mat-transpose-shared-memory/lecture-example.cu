#include <stdio.h>
#include "../timerc.h"

#define DIM 32
#define REP 4

void __global__ copy_kernel_y_first(int *a, int *b, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = ix * n + iy;

    a[r] = b[r];
}

void __global__ copy_kernel_x_first(int *a, int *b, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = iy * n + ix;

    a[r] = b[r];
}

void __global__ copy_kernel_x_first_multiple_elements(int *a, int *b, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = REP * iy * n + ix;

    for (int j = r; j < r + REP * n; j = j + n) {
        a[j] = b[j];
    }
}

void __global__ transpose_kernel_write_coalesced(int *a, int *b, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = iy * n + ix;
    int s = ix * n + iy;

    a[r] = b[s];
}

void __global__ transpose_kernel_write_coalesced_multi_ele(int *a, int *b, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = REP * iy * n + ix;
    int s = ix * n + REP * iy;

    for (int j = 0; j < REP; j++) {
        a[r + j * n] = b[s + j];
    }
}

void __global__ transpose_kernel_read_coalesced_multi_ele(int *a, int *b, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int r = REP * iy * n + ix;
    int s = ix * n + iy * REP;

    for (int j = 0; j < REP; j++) {
        a[s + j] = b[r + n * j];
    }
}

void __global__ transpose_kernel_read_coalesced(int *a, int *b, int n) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int ix2 = threadIdx.x + blockIdx.y * blockDim.y;
    int iy2 = threadIdx.y + blockIdx.x * blockDim.x;

    int r = iy * n + ix;
    int s = iy2 * n + ix2;

    __shared__ int tmpmat[DIM * DIM];  // 1D shared memory

    tmpmat[threadIdx.y + threadIdx.x * DIM] = b[r];

    __syncthreads();  // forces all threads inside the block to reach here. All memory transactions are visible as completed to other threads

    a[s] = tmpmat[threadIdx.x + threadIdx.y * DIM];
}

int main() {
    int n = DIM * 256;

    int *h_a = (int *)malloc(n * n * sizeof(int));
    int *h_b = (int *)malloc(n * n * sizeof(int));

    for (int i = 0; i < n * n; i++) {
        h_b[i] = i;
        h_a[i] = -1;
    }

    float cpu_time;
    cstart();

    for (int i = 0; i < n * n; i++) {
        h_a[i] = h_b[i];
    }

    cend(&cpu_time);

    printf("CPU copy time  = %f\n", cpu_time);

    int *h_d_a;
    int *h_d_b;

    cudaMalloc((void **)&h_d_a, n * n * sizeof(int));
    cudaMalloc((void **)&h_d_b, n * n * sizeof(int));

    cudaMemcpy(h_d_b, h_b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numthreadsperblock(DIM, DIM);
    dim3 numthreadsperblock2(DIM, DIM / REP);
    dim3 numblockspergrid(n / DIM, n / DIM);

    float gpu_time;
    gstart();
    //copy_kernel_y_first<<<numblockspergrid,numthreadsperblock>>>(h_d_a, h_d_b,  n);
    //copy_kernel_x_first<<<numblockspergrid,numthreadsperblock>>>(h_d_a, h_d_b,  n);
    copy_kernel_x_first_multiple_elements<<<numblockspergrid, numthreadsperblock2>>>(h_d_a, h_d_b, n);
    gend(&gpu_time);

    printf("GPU copy time  = %f\n", gpu_time);

    int *tmp = (int *)malloc(n * n * sizeof(int));
    cudaMemcpy(tmp, h_d_a, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * n; i++) {
        if (tmp[i] != h_a[i]) {
            printf("ERROR\n");
            break;
        }
    }
    printf("ALL GOOD. SO FAR... \n");

    cstart();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_a[j + n * i] = h_b[i + n * j];
            //h_a[i + n*j] = h_b[j + n*i]; // THIS IS 4 times SLOWER. Why?
        }
    }
    cend(&cpu_time);

    printf("CPU transpose time  = %f\n", cpu_time);

    gstart();
    //transpose_kernel_write_coalesced<<<numblockspergrid,numthreadsperblock>>>(h_d_a, h_d_b,  n);
    transpose_kernel_write_coalesced_multi_ele<<<numblockspergrid, numthreadsperblock2>>>(h_d_a, h_d_b, n);
    gend(&gpu_time);

    printf("GPU transpose write_coalesced time  = %f\n", gpu_time);

    gstart();
    //transpose_kernel_read_coalesced<<<numblockspergrid,numthreadsperblock>>>(h_d_a, h_d_b,  n);
    transpose_kernel_read_coalesced_multi_ele<<<numblockspergrid, numthreadsperblock2>>>(h_d_a, h_d_b, n);
    gend(&gpu_time);
    printf("GPU transpose read_coalesced time  = %f\n", gpu_time);

    free(tmp);
    free(h_a);
    free(h_b);
    cudaFree(h_d_a);
    cudaFree(h_d_b);

    return 0;
}
