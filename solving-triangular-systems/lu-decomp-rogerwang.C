/*--------------------------------------------------------------
 *    LU Decomposition Solver
 *    Finds LU Decomp of a 5x5 matrix
 *    Can used to solver linear system, get mat determinant
 *    and mat inverse.
 *-------------------------------------------------------------- */

#include <stdio.h>
#include <iomanip>
#include <iostream>
// const int n = 3;
const int n = 5;

void printHeader(int assignment, const char* date) {
    printf("==========================================\n");
    printf("Numerical Methods & Scientific Computing\nAssignment %d\n", assignment);
    printf("Author: Roger Wang\n");
    printf("Date: %s\n", date);
    printf("==========================================\n\n");
}

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

void LUDecomposition(double a[n][n], double lower[n][n], double upper[n][n]) {
    for (int i = 0; i < n; i++) {
        // Upper Triangular
        for (int k = i; k < n; k++) {
            // Summation of L(i, j) * U(j, k)
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (lower[i][j] * upper[j][k]);

            // Evaluating U(i, k)
            upper[i][k] = a[i][k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < n; k++) {
            if (i == k)
                lower[i][i] = 1;  // Diagonal as 1
            else {
                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (lower[k][j] * upper[j][i]);

                // Evaluating L(k, i)
                lower[k][i] = (a[k][i] - sum) / upper[i][i];
            }
        }
    }
}

void Crout(double Mat[n][n]) {
    int j, i, k;
    for (j = 0; j < n; j++) {
        for (i = 0; i <= j; i++) {
            if (i > 0) {
                for (k = 0; k < i; k++) {
                    Mat[i][j] -= Mat[i][k] * Mat[k][j];
                }
            }
        }
        if (j < n - 1) {
            for (i = j + 1; i < n; i++) {
                if (j > 0) {
                    for (k = 0; k < j; k++) {
                        Mat[i][j] -= Mat[i][k] * Mat[k][j];
                    }
                }
                Mat[i][j] = Mat[i][j] / Mat[j][j];
            }
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

void BackSub(double A[n][n], double y[n], double x[n]) {
    x[n - 1] = y[n - 1] / A[n - 1][n - 1];
    for (int k = n - 2; k >= 0; k--) {
        x[k] = y[k];
        for (int j = k + 1; j < n; j++) {
            x[k] = x[k] - A[k][j] * x[j];
        }
        x[k] = x[k] / A[k][k];
    }
}

void InvertMat(double LU[n][n], double out[n][n]) {
    double b[n], y[n], x[n];
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            b[j] = 0.0;
        }
        b[k] = 1.0;
        ForwardSub(LU, b, y);
        BackSub(LU, y, x);
        for (int j = 0; j < n; j++) {
            out[j][k] = x[j];
        }
    }
}

double Det(double A[n][n]) {
    double det = A[0][0];
    for (int i = 1; i < n; i++) {
        det = det * A[i][i];
    }
    return det;
}

int main() {
    printHeader(2, "2/24/20");
    double b[n] = {41.0, -9, 8.0, 3.7, -7.7};
    double a[n][n] = {{1.0, 7.0, 18.0, 4, 0.0},
                      {6.0, 0.1, 7.0, 0.0, 2.0},
                      {0.0, -5.0, -6.0, 4.0, 3.0},
                      {19.3, 3.0, -0.2, -0.4, -5.0},
                      {7.3, -2.0, 17.2, 5.4, 0.2}};
    double a_old[n][n];
    copyMat(a, a_old);

    printf("Initial Matrix A: \n");
    printMat(a);
    printf("After Crouts: \n");
    Crout(a);
    printMat(a);
    double l[n][n], u[n][n], res[n][n];
    LUDecomposition(a_old, l, u);
    printf("TEST: L x U (should equal A): \n");
    MatMultiply(l, u, res);
    printMat(res);

    double y[n], x[n];
    printf("Initial Vector b: \n");
    printVec(b);
    printf("Forward Sub:\n");
    ForwardSub(a, b, y);
    printVec(y);
    printf("Back Sub:\n");
    BackSub(a, y, x);
    printVec(x);

    printf("Inverted Mat:\n");
    double invA[n][n];
    InvertMat(a, invA);
    printMat(invA);
    printf("Test: A * invA: (should equal Identity Mat)\n");
    MatMultiply(a_old, invA, res);
    printMat(res);

    double det;
    det = Det(a);
    printf("Determinant: %f\n", det);

    return 0;
}
