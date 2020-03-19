#include <stdio.h>
#include "../timerc.h"

__global__ void warmup() {
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

__global__ void gpu_divergent_fixed() {
    int s = 1 - 2 * (threadIdx.x / 16);

    double v = 0;
    for (int i = 0; i < 1000; i++) {
        for (int j = i; j < i * i; j++) {
            v = v + s * i + i * j / 3.435;
        }
    }
}

void cpu() {
    double v = 0;
    for (int i = 0; i < 1000; i++) {
        for (int j = i; j < i * i; j++) {
            v = v + i + i * j / 3.435;
        }
    }
}

int main() {
    cudaSetDevice(1);

    warmup<<<1, 1>>>();

    float GPUtime;
    gstart();
    gpu<<<1, 32>>>();
    gend(&GPUtime);
    printf("GPU time no divergence = %f\n", GPUtime);

    gstart();
    gpu_divergent<<<1, 32>>>();
    gend(&GPUtime);
    printf("GPU time WITH divergence = %f\n", GPUtime);

    gstart();
    gpu_divergent_fixed<<<1, 32>>>();
    gend(&GPUtime);
    printf("GPU time WITH divergence FIXED = %f\n", GPUtime);

    float CPUtime;
    cstart();
    cpu();
    cend(&CPUtime);

    printf("CPU time = %f\n", CPUtime);

    return 0;
}
