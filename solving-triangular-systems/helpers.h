#ifndef HELPERS_DOT_H
#define HELPERS_DOT_H

__host__ __device__ void printVec(double *b, int n, const char *title);
__host__ __device__ void printMat(double *a, int n, const char *title);
__host__ __device__ void printSubMat(double *a, int idx, int size, int n, const char *title);
void multMatVec(double *a, double *b, double *x, int n);
double *initMat(int n);
double *initVec(int n);
void copyVec(double *newB, double *b, int n);
__host__ __device__ void copyMat(double *newA, double *a, int idx, int size, int n);
void matMult(double *mat1, double *mat2, double *res, int n);
__host__ __device__ void multSubMat(double *a, int idx_1, int idx_2, int size, int n, double *res);
__host__ __device__ void addNegative(double *a, int idx, int size, int n);
__host__ __device__ void invertTwoByTwo(double *a, int idx, int n);

#endif