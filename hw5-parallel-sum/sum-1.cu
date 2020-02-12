#include<stdio.h>
#include "../timerc.h"

__global__ void sum1(int *a, int * b, int n, int c){
	int ix = threadIdx.x + blockIdx.x*blockDim.x; // 1D block, 1D grid
	int local_total = 0;
	if (c*ix < n){
		int limit = c*ix +(c-1);
		if (limit >= n) {
			limit = n - 1;
		}
		for (int i = c*ix; i<= limit; i++) {
			local_total += a[i];
		}
		//a[c*ix] = local_total;
		b[ix] = local_total;
	}


}

__global__ void sum2(int *in, int * out, int size){

	int * shifted_in = in + blockIdx.x*2*blockDim.x;
	int local_ix = threadIdx.x;

	for (int s = 1; s <= blockDim.x; s = s*2){

		if (local_ix < blockDim.x/s){
			shifted_in[ 2*s*local_ix  ] = shifted_in[ 2*s*local_ix  ] +  shifted_in[ 2*s*local_ix  + s];

		}

		__syncthreads(); // does not sync threads in different blocks
	}

	if (local_ix == 0){

		out[blockIdx.x] = shifted_in[ 0 ];
	}

}

int main(){

	int size = 64*1024*1024;
	int c = 64;
	int blocksize = 1024;
        int gridsize = size/(2*blocksize);

	int *v = (int *)malloc(size * sizeof(int));
	int *h_d_v;
	int *h_d_o;
	cudaMalloc((void **)&h_d_v, size*sizeof(int));
	//cudaMalloc((void **)&h_d_o, (size/c)*sizeof(int));
	cudaMalloc((void **)&h_d_o, (gridsize)*sizeof(int));


	for (int i=0; i<size; i++) {
		v[i] = 1;
	}

	int total = 0;

	float cpu_time;
	cstart();
	for (int i=0; i<size; i++) {
		total += v[i];
	}
	cend(&cpu_time);
	printf("total = %d, cpu time = %f\n", total, cpu_time);

	cudaMemcpy(h_d_v, v, size*sizeof(int), cudaMemcpyHostToDevice);


	float gpu_time;

	gstart();
	//sum1<<<1024, 1024>>>(h_d_v, h_d_o, size, c);
	sum2<<<gridsize, blocksize>>>(h_d_v, h_d_o, size);
	gend(&gpu_time);
	printf("gpu kernel time = %f\n", gpu_time);

	gstart();
	//cudaMemcpy(v, h_d_o, (size/c)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(v, h_d_o, (gridsize)*sizeof(int), cudaMemcpyDeviceToHost);


	gend(&gpu_time);

        printf("gpu copy dev to host time = %f\n", gpu_time);

	cstart();
	int gpu_total = 0;
	//for (int i=0; i<(size/c); i++) {
	for (int i=0; i < gridsize; i++) {
		//printf("%d ",v[i]);
		gpu_total += v[i];
	}
	cend(&cpu_time);
        printf("cpu finish gpu sum time = %f\n", cpu_time);

	printf("gpu total = %d\n",gpu_total);


	free(v);
	cudaFree(h_d_v);

	return 0;
}
