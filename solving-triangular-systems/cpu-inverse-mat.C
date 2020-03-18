#include <stdio.h>
#include <stdlib.h>

int N = 8;

double **initMat(int n) {
    double **a = (double **)malloc(n * sizeof(double *));
    int c = 1;
    for (int i = 0; i < n; ++i) {
        a[i] = (double *)malloc(n * sizeof(double));
        for (int j = 0; j < n; ++j) {
            a[i][j] = c++;
        }
    }
    return a;
}

void printSubMat(double **a, int idx, int size, int n) {
    printf("%d %d\n\n", idx, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int local_idx = idx + i * n + j;
            printf("%d ", local_idx);
            printf("%f ", *a[local_idx]);
        }
        printf("\n");
    }
}

void freeMat(double **a, int n) {
    for (int i = 0; i < n; ++i) {
        free(a[i]);
    }
    free(a);
}

void printMat(double **a, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%4.1f ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void invertTwoByTwo(double **a, int n) {
    double det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
    printf("%f\n", det);
    double temp;
    temp = a[1][1];
    a[1][1] = a[0][0];
    a[0][0] = temp;
    a[1][0] = -a[1][0];
    a[0][1] = -a[0][1];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = (1 / det) * a[i][j];
        }
    }
}

// void inverseRecurse(double **a, int low, int high) {
//     int n = high - low;
//     if (n <= 2) {
//         invertTwoByTwo(a[], n);
//     }
//     inverseRecurse()
// }

int main() {
    int n = 4;
    double **a;
    a = initMat(n);
    printMat(a, n);

    printSubMat(a, n / 2, n / 2, n);
    freeMat(a, n);
    return 0;
}
