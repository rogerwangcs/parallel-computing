/*--------------------------------------------------------------
 *    CPU Forward Sub
 *-------------------------------------------------------------- */

#include <stdio.h>
#include <iomanip>
#include <iostream>
const int n = 3;

void printMat(double a[n][n]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%10.6f ", a[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printVec(double vec[n]) {
    for (int i = 0; i < n; i++) {
        printf("%10.6f ", vec[i]);
    }
    printf("\n");
}

// multiple two n x n matrices and put result in res
void MatMultiply(double mat1[n][n], double mat2[n][n], double res[n][n]) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            res[i][j] = 0.0;
            for (k = 0; k < n; k++)
                res[i][j] += mat1[i][k] * mat2[k][j];
        }
    }
}

void multiplyVec(double mat[n][n], double vec[n], double res[n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i] += mat[i][j] * vec[j];
        }
    }
}

// s: source, d: destination
void copyMat(double s[n][n], double d[n][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            d[i][j] = s[i][j];
        }
    }
}

void ForwardSub(double A[n][n], double b[n], double y[n]) {
    for (int k = 0; k < n; k++) {
        y[k] = b[k];
        for (int j = 0; j < k; j++) {
            y[k] = y[k] - A[k][j] * y[j];
        }
    }
}

int main() {
    double b[n] = {5, 5, 5};
    double y[n];
    double a[n][n] = {{1, 0, 0},
                      {0, 1, 0},
                      {0, 0, 1}};
    double a_old[n][n];
    copyMat(a, a_old);

    printf("Initial Matrix A: \n");
    printMat(a);

    ForwardSub(a, b, y);
    printf("Forward Sub:\n");
    printVec(y);

    return 0;
}
