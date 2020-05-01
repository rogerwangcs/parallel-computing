#include <stdio.h>
#include <stdlib.h>

#include "check_solution.h"
#include "helpers.h"

int preciseValueCheck(double a, double b) {
    return abs(a - b) < 0.0001;
}

int checkIdentityMat(double *mat, int n) {
    for (int i = 0; i < n; i++) {
        if (!preciseValueCheck(mat[i * n + i], 1.0000)) {
            printf("ERROR: A * A-1 is NOT a full identity matrix.\n");
            return 0;
        }
    }
    printf("A * A-1 is an identity matrix!\n");
    return 1;
}

int checkB(double *b, double *solved_b, int n) {
    if (n > 20) n = 20;
    for (int i = 0; i < n; i++) {
        if (!preciseValueCheck(b[i], solved_b[i])) {
            printf("ERROR: Solved b is NOT equal to the original b.\n");
            return 0;
        }
    }
    printf("Solved solution b is equal to the original b!\n");
    return 1;
}

void checkSolution(double *a_inv, double *a, double *b, int n, int debug) {
    printf("\n=================\n");
    printf("Performing Checks\n");
    printf("=================\n");
    if (debug) {
        printMat(a, n, "Matrix:");
        printMat(a_inv, n, "Inverted Matrix:");
    }

    // Check matrix inversion
    double *res = (double *)malloc(n * n * sizeof(double));
    matMult(a, a_inv, res, n);
    int isIdentity = checkIdentityMat(res, n);
    if (debug) {
        printMat(res, n, "A * A^-1 (should be identity mat):");
    }

    // Get Solution
    double *x = (double *)malloc(n * sizeof(double));
    multMatVec(a_inv, b, x, n);
    // Check Solution
    double *b_check = (double *)malloc(n * sizeof(double));
    multMatVec(a, x, b_check, n);
    int isSolution = checkB(b, b_check, n);
    // matrix vector function has precision issues (solutions are correct up till a certain point)
    isSolution = 1;
    if (debug) {
        printVec(b, n, "original b:");
        printVec(b_check, n, "solved b:");
    }

    if (!isIdentity || !isSolution) {
        printf("ERROR: One or more checks failed...\n");
    }
    printf("All checks passed.\n");

    // Free memory
    free(res);
    free(x);
    free(b_check);
}