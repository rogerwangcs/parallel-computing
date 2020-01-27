
// Roger Wang
// Parallel Computing
// Assignment 1

#include <stdio.h>
#include "../timerc.h"

__global__ void warmup() {
}

__global__ void helloKernel2d() {
    printf("I am a kernel running on block [%d, %d], thread [%d, %d] \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

__global__ void helloKernel3d() {
    printf("I am a kernel running on block [%d, %d, %d], thread [%d, %d, %d] \n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

void cpu() {
    double v = 0;
    for (int i = 0; i < 1000; i++) {
        for (int j = i; j < i * i; j++) {
            v = v + i + i * j / 3.435;
        }
    }
}

__global__ void gpu() {
    double v = 0;
    for (int i = 0; i < 1000; i++) {
        for (int j = i; j < i * i; j++) {
            v = v + i + i * j / 3.435;
        }
    }
}

__global__ void gpu_divergent() {
    if (threadIdx.x < 16) {
        double v = 0;
        for (int i = 0; i < 1000; i++) {
            for (int j = i; j < i * i; j++) {
                v = v + i + i * j / 3.435;
            }
        }
    } else {
        double v = 0;
        for (int i = 0; i < 1000; i++) {
            for (int j = i; j < i * i; j++) {
                v = v - i + i * j / 3.435;
            }
        }
    }
}

int main() {
    cudaSetDevice(1);
    warmup<<<1, 1>>>();
    helloKernel2d<<<{2, 2}, {2, 2}>>>();
    helloKernel3d<<<{2, 2, 2}, {2, 2, 2}>>>();
    cudaDeviceSynchronize();

    float GPUtime;
    gstart();
    gpu<<<1, 32>>>();
    gend(&GPUtime);
    printf("GPU time no divergence = %f\n", GPUtime);

    gstart();
    gpu_divergent<<<1, 32>>>();
    gend(&GPUtime);
    printf("GPU time WITH divergence = %f\n", GPUtime);

    float CPUtime;
    cstart();
    cpu();
    cend(&CPUtime);
    printf("CPU time = %f\n", CPUtime);

    printf("\n");
    return 0;
}
