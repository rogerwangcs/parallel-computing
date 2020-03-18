// Notes:
// I did not have the foresight that this is due today during the break. I'm not gonna bs this one,
// I'll take a look and finish this code by the end of the break.
// Thanks! Roger

#include <stdio.h>
#include "../timerc.h"

#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__host__ __device__ inline int pos_mod(int s, int m)
{
	if (s >= 0){
		return s % m;
	}
	else{
		return m +  (s % m);
	}
}


__host__ __device__ int countN(int *game, int x, int y, int dim)
{

	int n = 0;
	int xp1 = pos_mod(x+1,dim);
        int xm1 = pos_mod(x-1,dim);
        int yp1 = pos_mod(y+1,dim);
        int ym1 = pos_mod(y-1,dim);

	n = game[y*dim + xm1] + game[y*dim + xp1] + game[ym1*dim + xm1] +
		game[ym1*dim + xp1] + game[ym1*dim + x] + game[yp1*dim + xm1] +
		game[yp1*dim + xp1] + game[yp1*dim + x];

	return n;

}


void play_game_cpu(int * game_new, int * game_old, int dim){

	for (int y = 0; y < dim; y++){
		for (int x = 0; x < dim; x++){

			game_new[y*dim + x] = game_old[y*dim + x];

			int n = countN(game_old, x, y, dim);

			if ( game_old[y*dim + x]  == 1 && (n < 2 || n > 3)  ){
				game_new[y*dim + x] = 0;
			}

			if ( game_old[y*dim + x] == 1 && (n == 2 || n == 3)){
				game_new[y*dim + x] = 1;
			}

			if ( game_old[y*dim + x] == 0 && n == 3  ){
				game_new[y*dim + x] = 1;
			}

		}
	}
}


__global__ void play_game_gpu(int * game_new, int * game_old, int dim){

       		int y = threadIdx.y + blockIdx.y * blockDim.y;
       		int x = threadIdx.x + blockIdx.x * blockDim.x;

		if (x < dim && y < dim){


                        game_new[y*dim + x] = game_old[y*dim + x];

                        int n = countN(game_old, x, y, dim);

                        if ( game_old[y*dim + x]  == 1 && (n < 2 || n > 3)  ){
                                game_new[y*dim + x] = 0;
                        }

                        if ( game_old[y*dim + x] == 1 && (n == 2 || n == 3)){
                                game_new[y*dim + x] = 1;
                        }

                        if ( game_old[y*dim + x] == 0 && n == 3  ){
                                game_new[y*dim + x] = 1;
                        }

		}



}

texture<int> tex_game_1;
texture<int> tex_game_2;

texture<int,2> tex2D_game_1;
texture<int,2> tex2D_game_2;

texture<int,2> tex2D_cuarr_game_1;
texture<int,2> tex2D_cuarr_game_2;

surface<void,2> surf_1;
surface<void,2> surf_2;

__global__ void tex1D_play_game_gpu(int * game_new , int dir, int dim  ){
 	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x < dim && y < dim){

	   	int xp1 = pos_mod(x+1,dim);
        	int xm1 = pos_mod(x-1,dim);
        	int yp1 = pos_mod(y+1,dim);
        	int ym1 = pos_mod(y-1,dim);

		int n;
		int old_val;

		if (dir == 1){
			old_val = tex1Dfetch(tex_game_1, x + y*dim);
			game_new[x + y*dim] = old_val;

			n = tex1Dfetch(tex_game_1, xm1 + y*dim) + tex1Dfetch(tex_game_1, xp1 + y*dim) + tex1Dfetch(tex_game_1, x + yp1*dim) + tex1Dfetch(tex_game_1, x + ym1*dim) + tex1Dfetch(tex_game_1, xm1 + ym1*dim) + tex1Dfetch(tex_game_1, xp1 + yp1*dim) + tex1Dfetch(tex_game_1, xm1 + yp1*dim) + tex1Dfetch(tex_game_1, xp1 + ym1*dim);

		}else{

  			old_val = tex1Dfetch(tex_game_2, x + y*dim);
                        game_new[x + y*dim] = old_val;

                        n = tex1Dfetch(tex_game_2, xm1 + y*dim) + tex1Dfetch(tex_game_2, xp1 + y*dim) + tex1Dfetch(tex_game_2, x + yp1*dim) + tex1Dfetch(tex_game_2, x + ym1*dim) + tex1Dfetch(tex_game_2, xm1 + ym1*dim) + tex1Dfetch(tex_game_2, xp1 + yp1*dim) + tex1Dfetch(tex_game_2, xm1 + yp1*dim) + tex1Dfetch(tex_game_2, xp1 + ym1*dim);

		}

                if ( old_val  == 1 && (n < 2 || n > 3)  ){
	                game_new[y*dim + x] = 0;
                }

                if ( old_val == 1 && (n == 2 || n == 3)){
                	game_new[y*dim + x] = 1;
                }

                if ( old_val == 0 && n == 3  ){
                        game_new[y*dim + x] = 1;
                }


	}

}


__global__ void tex2D_play_game_gpu(int * game_new , int dir, int dim  ){
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x < dim && y < dim){

                int xp1 = pos_mod(x+1,dim);
                int xm1 = pos_mod(x-1,dim);
                int yp1 = pos_mod(y+1,dim);
                int ym1 = pos_mod(y-1,dim);

                int n;
                int old_val;

                if (dir == 1){
                        old_val = tex2D(tex2D_game_1, x , y);
                        game_new[x + y*dim] = old_val;

                        n = tex2D(tex2D_game_1, xm1 ,  y) + tex2D(tex2D_game_1, xp1 , y) + tex2D(tex2D_game_1, x , yp1) + tex2D(tex2D_game_1, x , ym1) + tex2D(tex2D_game_1, xm1 , ym1) + tex2D(tex2D_game_1, xp1 , yp1) + tex2D(tex2D_game_1, xm1 , yp1) + tex2D(tex2D_game_1, xp1 , ym1);

                }else{

                        old_val = tex2D(tex2D_game_2, x , y);
                        game_new[x + y*dim] = old_val;

                        n = tex2D(tex2D_game_2, xm1 , y) + tex2D(tex2D_game_2, xp1 , y) + tex2D(tex2D_game_2, x , yp1) + tex2D(tex2D_game_2, x , ym1) + tex2D(tex2D_game_2, xm1 , ym1) + tex2D(tex2D_game_2, xp1 , yp1) + tex2D(tex2D_game_2, xm1 , yp1) + tex2D(tex2D_game_2, xp1 , ym1);

                }

                if ( old_val  == 1 && (n < 2 || n > 3)  ){
                        game_new[y*dim + x] = 0;
                }

                if ( old_val == 1 && (n == 2 || n == 3)){
                        game_new[y*dim + x] = 1;
                }

                if ( old_val == 0 && n == 3  ){
                        game_new[y*dim + x] = 1;
                }


        }

}




__global__ void tex2D_cuarr_play_game_gpu(int * game_new , int dir, int dim  ){
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x < dim && y < dim){

                int xp1 = pos_mod(x+1,dim);
                int xm1 = pos_mod(x-1,dim);
                int yp1 = pos_mod(y+1,dim);
                int ym1 = pos_mod(y-1,dim);

                int n;
                int old_val;

                if (dir == 1){
                        old_val = tex2D(tex2D_cuarr_game_1, x , y);
                        game_new[x + y*dim] = old_val;

                        n = tex2D(tex2D_cuarr_game_1, xm1 ,  y) + tex2D(tex2D_cuarr_game_1, xp1 , y) + tex2D(tex2D_cuarr_game_1, x , yp1) + tex2D(tex2D_cuarr_game_1, x , ym1) + tex2D(tex2D_cuarr_game_1, xm1 , ym1) + tex2D(tex2D_cuarr_game_1, xp1 , yp1) + tex2D(tex2D_cuarr_game_1, xm1 , yp1) + tex2D(tex2D_cuarr_game_1, xp1 , ym1);

                }else{

                        old_val = tex2D(tex2D_cuarr_game_2, x , y);
                        game_new[x + y*dim] = old_val;

                        n = tex2D(tex2D_cuarr_game_2, xm1 , y) + tex2D(tex2D_cuarr_game_2, xp1 , y) + tex2D(tex2D_cuarr_game_2, x , yp1) + tex2D(tex2D_cuarr_game_2, x , ym1) + tex2D(tex2D_cuarr_game_2, xm1 , ym1) + tex2D(tex2D_cuarr_game_2, xp1 , yp1) + tex2D(tex2D_cuarr_game_2, xm1 , yp1) + tex2D(tex2D_cuarr_game_2, xp1 , ym1);

                }

                if ( old_val  == 1 && (n < 2 || n > 3)  ){
                        game_new[y*dim + x] = 0;
                }

                if ( old_val == 1 && (n == 2 || n == 3)){
                        game_new[y*dim + x] = 1;
                }

                if ( old_val == 0 && n == 3  ){
                        game_new[y*dim + x] = 1;
                }


        }

}


__global__ void tex2D_cuarr_surf_play_game_gpu( int dir, int dim  ){
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        if (x < dim && y < dim){

                int xp1 = pos_mod(x+1,dim);
                int xm1 = pos_mod(x-1,dim);
                int yp1 = pos_mod(y+1,dim);
                int ym1 = pos_mod(y-1,dim);

                int n;
                int old_val;

                if (dir == 1){
                        old_val = tex2D(tex2D_cuarr_game_1, x , y);
                        surf2Dwrite( old_val , surf_2 , 4*x , y , cudaBoundaryModeTrap );
			//game_new[x + y*dim] = old_val;

                        n = tex2D(tex2D_cuarr_game_1, xm1 ,  y) + tex2D(tex2D_cuarr_game_1, xp1 , y) + tex2D(tex2D_cuarr_game_1, x , yp1) + tex2D(tex2D_cuarr_game_1, x , ym1) + tex2D(tex2D_cuarr_game_1, xm1 , ym1) + tex2D(tex2D_cuarr_game_1, xp1 , yp1) + tex2D(tex2D_cuarr_game_1, xm1 , yp1) + tex2D(tex2D_cuarr_game_1, xp1 , ym1);


		        if ( old_val  == 1 && (n < 2 || n > 3)  ){
                     		surf2Dwrite( 0 , surf_2 , 4*x , y , cudaBoundaryModeTrap );

                	}

                	if ( old_val == 1 && (n == 2 || n == 3)){
                        	surf2Dwrite( 1 , surf_2 , 4*x , y , cudaBoundaryModeTrap );
                	}

                	if ( old_val == 0 && n == 3  ){
                               surf2Dwrite( 1 , surf_2 , 4*x , y , cudaBoundaryModeTrap );
                	}



		}else{

                        old_val = tex2D(tex2D_cuarr_game_2, x , y);
                        surf2Dwrite( old_val , surf_1 , 4*x , y , cudaBoundaryModeTrap );


                        n = tex2D(tex2D_cuarr_game_2, xm1 , y) + tex2D(tex2D_cuarr_game_2, xp1 , y) + tex2D(tex2D_cuarr_game_2, x , yp1) + tex2D(tex2D_cuarr_game_2, x , ym1) + tex2D(tex2D_cuarr_game_2, xm1 , ym1) + tex2D(tex2D_cuarr_game_2, xp1 , yp1) + tex2D(tex2D_cuarr_game_2, xm1 , yp1) + tex2D(tex2D_cuarr_game_2, xp1 , ym1);


			if ( old_val  == 1 && (n < 2 || n > 3)  ){
                                  surf2Dwrite( 0 , surf_1 , 4*x , y , cudaBoundaryModeTrap );
                	}

                	if ( old_val == 1 && (n == 2 || n == 3)){
                                   surf2Dwrite( 1 , surf_1 , 4*x , y , cudaBoundaryModeTrap );
                	}

                	if ( old_val == 0 && n == 3  ){
                                    surf2Dwrite( 1 , surf_1 , 4*x , y , cudaBoundaryModeTrap );
                	}

                }





        }

}



void setrandomconfig(int *game, int dim, float p){

	for (int i = 0; i < dim*dim; i++){
		game[i] = ((double) rand() / RAND_MAX) < p;
	}

}





int main(){

	int dim = 1024;
	int size = dim*dim;
	int num_iter = 100;

	int * h_game_1 = (int *) malloc(size * sizeof(int));
	int * h_game_2 = (int *) malloc(size * sizeof(int));

     	int * d_game_1;
        int * d_game_2;
        cudaMalloc( (void**)& d_game_1  , sizeof(int) * size   );
        cudaMalloc( (void**)& d_game_2 , sizeof(int) * size   );

	cudaArray * cuarr_game_1;
	cudaArray * cuarr_game_2;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>(); //constructor


	cudaMallocArray( &cuarr_game_1  , &desc, dim, dim , cudaArraySurfaceLoadStore  );
        cudaMallocArray( &cuarr_game_2 ,  &desc, dim, dim , cudaArraySurfaceLoadStore  );


	setrandomconfig(h_game_1, dim, 0.6);

        cudaMemcpy( d_game_1  , h_game_1   , sizeof(int)*size ,  cudaMemcpyHostToDevice);

	cudaMemcpyToArray(  cuarr_game_1 , 0 , 0 , h_game_1 , sizeof(int)*size ,  cudaMemcpyHostToDevice   );

	float cpu_time;
	cstart();
	for (int t = 1; t < num_iter/2; t++){
		play_game_cpu(h_game_2, h_game_1, dim);
                play_game_cpu(h_game_1, h_game_2, dim);
	}
	cend(&cpu_time);
	printf("CPU time for %d iterations = %f\n",num_iter,cpu_time);

	dim3 numthreadsperblock(32,32);
	dim3 numblockspergrid(32,32);

	cudaBindTexture(NULL, tex_game_1, d_game_1,size*sizeof(int));
	cudaBindTexture(NULL, tex_game_2, d_game_2,size*sizeof(int));


	cudaBindTexture2D(NULL, tex2D_game_1, d_game_1,desc, dim, dim, dim*sizeof(int));
        cudaBindTexture2D(NULL, tex2D_game_2, d_game_2,desc, dim, dim, dim*sizeof(int));

	cudaBindTextureToArray( tex2D_cuarr_game_1, cuarr_game_1 , desc   );
	cudaBindTextureToArray( tex2D_cuarr_game_2, cuarr_game_2 , desc   );


	cudaBindSurfaceToArray( surf_1 , cuarr_game_1 );
	cudaBindSurfaceToArray( surf_2 , cuarr_game_2 );


	float gpu_time;
	gstart();
	for (int t = 1; t < num_iter/2; t++){
		//play_game_gpu<<<numblockspergrid,numthreadsperblock>>>(d_game_2, d_game_1, dim);
		//play_game_gpu<<<numblockspergrid,numthreadsperblock>>>(d_game_1, d_game_2, dim);

		//tex1D_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>( d_game_2  , 1 ,  dim  );
		//tex1D_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>( d_game_1  , 2 ,  dim  );

		//tex2D_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>( d_game_2  , 1 ,  dim  );
                //tex2D_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>( d_game_1  , 2 ,  dim  );

		//tex2D_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>( d_game_2  , 1 ,  dim  );
                //cudaMemcpyToArray(  cuarr_game_2 , 0 , 0 , d_game_2 , sizeof(int)*size ,  cudaMemcpyDeviceToDevice   );
		//tex2D_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>( d_game_1  , 2 ,  dim  );
		//cudaMemcpyToArray(  cuarr_game_1 , 0 , 0 , d_game_1 , sizeof(int)*size ,  cudaMemcpyDeviceToDevice   );

		tex2D_cuarr_surf_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>(  1 ,  dim  );
                tex2D_cuarr_surf_play_game_gpu<<<numblockspergrid,numthreadsperblock>>>(  2 ,  dim  );

	}
	gend(&gpu_time);
        printf("GPU time for %d iterations = %f\n",num_iter,gpu_time);

	int * h_game_3 = (int *) malloc(size * sizeof(int));

	cudaMemcpy( h_game_3  , d_game_1   , sizeof(int)*size ,  cudaMemcpyDeviceToHost);
	cudaMemcpyFromArray( h_game_3 ,  cuarr_game_1 , 0 , 0  , sizeof(int)*size ,  cudaMemcpyDeviceToHost   );


	int f = 0;
	for (int i = 0; i < size; i++){
		if ( h_game_3[i] != h_game_1[i]  ){
			f = 1;
			break;
		}
	}
	if (f == 1){
		printf("ERRORS\n");
	}else{
		printf("ALL GOOD: NO ERRORS\n");
	}

    	gerror( cudaPeekAtLastError() );
	cudaDeviceSynchronize();

	cudaUnbindTexture(tex2D_game_1);
        cudaUnbindTexture(tex2D_game_2);
	cudaUnbindTexture(tex_game_1);
	cudaUnbindTexture(tex_game_2);
	cudaFree(d_game_1);
	cudaFree(d_game_2);
	free(h_game_1);
	free(h_game_2);
	free(h_game_3);


	return 0;
}
