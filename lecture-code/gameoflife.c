#include <stdio.h>

__host__ __device__ int inline pos_mod(int s, int m) {
    if (s >= 0) {
        return s % m;
    } else {
        return m + (s % m);
    }
}

__host__ __device__ int countN(int *game, int x, int y, int dim) {
    int n = 0;
    int xp1 = mod(x + 1, dim);
}

int main() {
    return 0;
}