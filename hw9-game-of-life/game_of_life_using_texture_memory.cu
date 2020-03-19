#include <stdio.h>
#include "../timerc.h"

texture<int> tex_game_1;
texture<int> tex_game_2;

texture<int, 2> tex2D_game_1;
texture<int, 2> tex2D_game_2;

texture<int, 2> tex2D_game_1_for_cuarray_use;
texture<int, 2> tex2D_game_2_for_cuarray_use;

surface<void, 2> outputSurface_game_1;
surface<void, 2> outputSurface_game_2;

#define gerror(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__host__ __device__ void printgame(int *game, int dim) {
    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            printf("%d ", game[y * dim + x]);
        }
        printf("\n");
    }
    printf("\n");
}

__host__ __device__ inline int positive_mod(int s, int m) {
    if (s >= 0) {
        return s % m;
    } else {
        return m + (s % m);
    }
}

__host__ __device__ int countneigh(int *game, int x, int y, int dim) {
    int n = 0;

    int xp1 = positive_mod(x + 1, dim);
    int xm1 = positive_mod(x - 1, dim);
    int yp1 = positive_mod(y + 1, dim);
    int ym1 = positive_mod(y - 1, dim);

    n = game[y * dim + xm1] +
        game[y * dim + xp1] +
        game[yp1 * dim + x] +
        game[ym1 * dim + x] +
        game[ym1 * dim + xm1] +
        game[yp1 * dim + xp1] +
        game[yp1 * dim + xm1] +
        game[ym1 * dim + xp1];

    return n;
}

// we are going to launch a 2D kerned.
__global__ void simple_play_game_gpu(int *game_new, int *game_old, int dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < dim && y < dim) {
        // first copy input to output. Then make transitions.
        game_new[y * dim + x] = game_old[y * dim + x];

        int num_neigh_cells = countneigh(game_old, x, y, dim);

        //game_new[y*dim + x] =num_neigh_cells;

        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (game_old[y * dim + x] == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3)) {
            game_new[y * dim + x] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (game_old[y * dim + x] == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3)) {
            game_new[y * dim + x] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (game_old[y * dim + x] == 0 && num_neigh_cells == 3) {
            game_new[y * dim + x] = 1;
        }
    }
}

// we are going to launch a 2D kerned.
// the flag into1into2 tells us which texture we should take as our starting texture
// note that the order of mapping threads to textures still affects speed.
// in particular, it is faster to have the fastest changing index, the threadIdx.x to be associated to the x component of the texture
// since in the device memory, the textures are addressed in row-major order, we should read them as tex2D(tex2D_game_1,xm1,y) and not tex2D(tex2D_game_1,xm1,y),
// assuming that xm1 is affected by threadIdx.x and that y is affected by threadIdx.y
__global__ void textures2D_play_game_gpu(int *destgame, int into1into2, int dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int curr_lin_ix = y * dim + x;

    if (x < dim && y < dim) {
        int xp1 = positive_mod(x + 1, dim);
        int xm1 = positive_mod(x - 1, dim);
        int yp1 = positive_mod(y + 1, dim);
        int ym1 = positive_mod(y - 1, dim);

        int num_neigh_cells;
        int old_value;

        if (into1into2 == 1) {
            old_value = tex2D(tex2D_game_1, x, y);

            destgame[curr_lin_ix] = old_value;

            num_neigh_cells = tex2D(tex2D_game_1, xm1, y) +
                              tex2D(tex2D_game_1, xp1, y) +
                              tex2D(tex2D_game_1, x, yp1) +
                              tex2D(tex2D_game_1, x, ym1) +
                              tex2D(tex2D_game_1, xm1, ym1) +
                              tex2D(tex2D_game_1, xp1, yp1) +
                              tex2D(tex2D_game_1, xm1, yp1) +
                              tex2D(tex2D_game_1, xp1, ym1);
        }
        if (into1into2 == 2) {
            old_value = tex2D(tex2D_game_2, x, y);
            destgame[curr_lin_ix] = old_value;

            num_neigh_cells = tex2D(tex2D_game_2, xm1, y) +
                              tex2D(tex2D_game_2, xp1, y) +
                              tex2D(tex2D_game_2, x, yp1) +
                              tex2D(tex2D_game_2, x, ym1) +
                              tex2D(tex2D_game_2, xm1, ym1) +
                              tex2D(tex2D_game_2, xp1, yp1) +
                              tex2D(tex2D_game_2, xm1, yp1) +
                              tex2D(tex2D_game_2, xp1, ym1);
        }

        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (old_value == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3)) {
            destgame[curr_lin_ix] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (old_value == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3)) {
            destgame[curr_lin_ix] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (old_value == 0 && num_neigh_cells == 3) {
            destgame[curr_lin_ix] = 1;
        }
    }
}

__global__ void textures2D_using_array_play_game_gpu(int *destgame, int into1into2, int dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int curr_lin_ix = y * dim + x;

    if (x < dim && y < dim) {
        int xp1 = positive_mod(x + 1, dim);
        int xm1 = positive_mod(x - 1, dim);
        int yp1 = positive_mod(y + 1, dim);
        int ym1 = positive_mod(y - 1, dim);

        int num_neigh_cells;
        int old_value;

        if (into1into2 == 1) {
            old_value = tex2D(tex2D_game_1_for_cuarray_use, x, y);

            destgame[curr_lin_ix] = old_value;

            num_neigh_cells = tex2D(tex2D_game_1_for_cuarray_use, xm1, y) +
                              tex2D(tex2D_game_1_for_cuarray_use, xp1, y) +
                              tex2D(tex2D_game_1_for_cuarray_use, x, yp1) +
                              tex2D(tex2D_game_1_for_cuarray_use, x, ym1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xm1, ym1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xp1, yp1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xm1, yp1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xp1, ym1);
        }
        if (into1into2 == 2) {
            old_value = tex2D(tex2D_game_2_for_cuarray_use, x, y);
            destgame[curr_lin_ix] = old_value;

            num_neigh_cells = tex2D(tex2D_game_2_for_cuarray_use, xm1, y) +
                              tex2D(tex2D_game_2_for_cuarray_use, xp1, y) +
                              tex2D(tex2D_game_2_for_cuarray_use, x, yp1) +
                              tex2D(tex2D_game_2_for_cuarray_use, x, ym1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xm1, ym1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xp1, yp1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xm1, yp1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xp1, ym1);
        }

        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (old_value == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3)) {
            destgame[curr_lin_ix] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (old_value == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3)) {
            destgame[curr_lin_ix] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (old_value == 0 && num_neigh_cells == 3) {
            destgame[curr_lin_ix] = 1;
        }
    }
}

__global__ void textures2D_using_array_and_surface_play_game_gpu(int into1into2, int dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < dim && y < dim) {
        int xp1 = positive_mod(x + 1, dim);
        int xm1 = positive_mod(x - 1, dim);
        int yp1 = positive_mod(y + 1, dim);
        int ym1 = positive_mod(y - 1, dim);

        int num_neigh_cells;
        int old_value;

        if (into1into2 == 1) {
            // write old value to surface memory that is bound to array_2
            old_value = tex2D(tex2D_game_1_for_cuarray_use, x, y);
            surf2Dwrite(old_value, outputSurface_game_2, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes

            num_neigh_cells = tex2D(tex2D_game_1_for_cuarray_use, xm1, y) +
                              tex2D(tex2D_game_1_for_cuarray_use, xp1, y) +
                              tex2D(tex2D_game_1_for_cuarray_use, x, yp1) +
                              tex2D(tex2D_game_1_for_cuarray_use, x, ym1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xm1, ym1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xp1, yp1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xm1, yp1) +
                              tex2D(tex2D_game_1_for_cuarray_use, xp1, ym1);

            //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
            //Any live cell with more than three live neighbours dies, as if by overpopulation.
            if (old_value == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3)) {
                surf2Dwrite(0, outputSurface_game_2, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes
            }
            //Any live cell with two or three live neighbours lives on to the next generation.
            if (old_value == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3)) {
                surf2Dwrite(1, outputSurface_game_2, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes
            }
            //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
            if (old_value == 0 && num_neigh_cells == 3) {
                surf2Dwrite(1, outputSurface_game_2, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes
            }
        }
        if (into1into2 == 2) {
            // write old value to surface memory that is bound to array_1
            old_value = tex2D(tex2D_game_2_for_cuarray_use, x, y);
            surf2Dwrite(old_value, outputSurface_game_1, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes

            num_neigh_cells = tex2D(tex2D_game_2_for_cuarray_use, xm1, y) +
                              tex2D(tex2D_game_2_for_cuarray_use, xp1, y) +
                              tex2D(tex2D_game_2_for_cuarray_use, x, yp1) +
                              tex2D(tex2D_game_2_for_cuarray_use, x, ym1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xm1, ym1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xp1, yp1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xm1, yp1) +
                              tex2D(tex2D_game_2_for_cuarray_use, xp1, ym1);

            //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
            //Any live cell with more than three live neighbours dies, as if by overpopulation.
            if (old_value == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3)) {
                surf2Dwrite(0, outputSurface_game_1, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes
            }
            //Any live cell with two or three live neighbours lives on to the next generation.
            if (old_value == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3)) {
                surf2Dwrite(1, outputSurface_game_1, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes
            }
            //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
            if (old_value == 0 && num_neigh_cells == 3) {
                surf2Dwrite(1, outputSurface_game_1, x * 4, y, cudaBoundaryModeTrap);  //note that the x component needs to be indexed in bytes
            }
        }
    }
}

// we are going to launch a 2D kerned.
// the flag into1into2 tells us which texture we should take as our starting texture
__global__ void textures1D_play_game_gpu(int *destgame, int into1into2, int dim) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int curr_lin_ix = y * dim + x;

    if (x < dim && y < dim) {
        int xp1 = positive_mod(x + 1, dim);
        int xm1 = positive_mod(x - 1, dim);
        int yp1 = positive_mod(y + 1, dim);
        int ym1 = positive_mod(y - 1, dim);

        int num_neigh_cells;
        int old_value;

        if (into1into2 == 1) {
            old_value = tex1Dfetch(tex_game_1, x + dim * y);
            destgame[curr_lin_ix] = old_value;

            num_neigh_cells =
                tex1Dfetch(tex_game_1, xm1 + y * dim) +
                tex1Dfetch(tex_game_1, xp1 + dim * y) +
                tex1Dfetch(tex_game_1, x + dim * yp1) +
                tex1Dfetch(tex_game_1, x + dim * ym1) +
                tex1Dfetch(tex_game_1, xm1 + dim * ym1) +
                tex1Dfetch(tex_game_1, xp1 + dim * yp1) +
                tex1Dfetch(tex_game_1, xm1 + dim * yp1) +
                tex1Dfetch(tex_game_1, xp1 + dim * ym1);
        }
        if (into1into2 == 2) {
            old_value = tex1Dfetch(tex_game_2, x + dim * y);
            destgame[curr_lin_ix] = old_value;

            num_neigh_cells =
                tex1Dfetch(tex_game_2, xm1 + dim * y) +
                tex1Dfetch(tex_game_2, xp1 + dim * y) +
                tex1Dfetch(tex_game_2, x + dim * yp1) +
                tex1Dfetch(tex_game_2, x + dim * ym1) +
                tex1Dfetch(tex_game_2, xm1 + dim * ym1) +
                tex1Dfetch(tex_game_2, xp1 + dim * yp1) +
                tex1Dfetch(tex_game_2, xm1 + dim * yp1) +
                tex1Dfetch(tex_game_2, xp1 + dim * ym1);
        }

        //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        //Any live cell with more than three live neighbours dies, as if by overpopulation.
        if (old_value == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3)) {
            destgame[curr_lin_ix] = 0;
        }
        //Any live cell with two or three live neighbours lives on to the next generation.
        if (old_value == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3)) {
            destgame[curr_lin_ix] = 1;
        }
        //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        if (old_value == 0 && num_neigh_cells == 3) {
            destgame[curr_lin_ix] = 1;
        }
    }
}

void setrandomconfi(int *game, int dim, float p) {
    for (int i = 0; i < dim * dim; i++) {
        game[i] = ((double)rand() / (RAND_MAX)) < p;
    }
}

void play_game_cpu(int *game_new, int *game_old, int dim) {
    // there order, either y first or x first, affects speed of the CPU code quite a bit
    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            // first copy input to output. Then make transitions.
            game_new[y * dim + x] = game_old[y * dim + x];

            int num_neigh_cells = countneigh(game_old, x, y, dim);

            //game_new[y*dim + x] =num_neigh_cells;

            //Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
            //Any live cell with more than three live neighbours dies, as if by overpopulation.
            if (game_old[y * dim + x] == 1 && (num_neigh_cells < 2 || num_neigh_cells > 3)) {
                game_new[y * dim + x] = 0;
            }
            //Any live cell with two or three live neighbours lives on to the next generation.
            if (game_old[y * dim + x] == 1 && (num_neigh_cells == 2 || num_neigh_cells == 3)) {
                game_new[y * dim + x] = 1;
            }
            //Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
            if (game_old[y * dim + x] == 0 && num_neigh_cells == 3) {
                game_new[y * dim + x] = 1;
            }
        }
    }
}

int main() {
    float cpu_time;
    float gpu_time;

    int gamedim = 1024;
    int gamesize = gamedim * gamedim;

    int num_iterations = 100;

    int *h_game_1 = (int *)malloc(sizeof(int) * gamesize);
    int *h_game_2 = (int *)malloc(sizeof(int) * gamesize);
    int *h_game_3 = (int *)malloc(sizeof(int) * gamesize);

    setrandomconfi(h_game_1, gamedim, 0.6);
    //printgame(h_game_1,gamedim);

    cudaSetDevice(0);
    cudaDeviceReset();

    // we create space and copy data to the GPU
    int *d_game_1;
    int *d_game_2;
    cudaMalloc((void **)&d_game_1, sizeof(int) * gamesize);
    cudaMalloc((void **)&d_game_2, sizeof(int) * gamesize);

    // here we bind the different textures
    // we use the same binding for the different textures we have at our disposal
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();

    // note that the pitch needs to be   gamedim     it cannot be    gamesize = gamedim * gamedim
    cudaBindTexture2D(NULL, tex2D_game_1, d_game_1, desc, gamedim, gamedim, sizeof(int) * gamedim);
    cudaBindTexture2D(NULL, tex2D_game_2, d_game_2, desc, gamedim, gamedim, sizeof(int) * gamedim);

    cudaBindTexture(NULL, tex_game_1, d_game_1, gamesize * sizeof(int));
    cudaBindTexture(NULL, tex_game_2, d_game_2, gamesize * sizeof(int));

    // create some array so that we can bind stuff to the array instead of device memory
    cudaArray *cuArray_game_1;
    cudaArray *cuArray_game_2;
    cudaMallocArray(&cuArray_game_1, &desc, gamedim, gamedim, cudaArraySurfaceLoadStore);
    cudaMallocArray(&cuArray_game_2, &desc, gamedim, gamedim, cudaArraySurfaceLoadStore);

    cudaBindSurfaceToArray(outputSurface_game_1, cuArray_game_1);  //note that, unlike textures, surfaces can only be bound to arrays !!
    cudaBindSurfaceToArray(outputSurface_game_2, cuArray_game_2);  //note that, unlike textures, surfaces can only be bound to arrays !!

    cudaBindTextureToArray(tex2D_game_1_for_cuarray_use, cuArray_game_1, desc);
    cudaBindTextureToArray(tex2D_game_2_for_cuarray_use, cuArray_game_2, desc);

    // now we copy stuff to the device memory and to the cuda array
    cudaMemcpy(d_game_1, h_game_1, sizeof(int) * gamesize, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuArray_game_1, 0, 0, h_game_1, sizeof(int) * gamesize, cudaMemcpyHostToDevice);

    dim3 numthreadsperblock(32, 32);
    dim3 numblocks = dim3((gamedim + 31) / 32, (gamedim + 31) / 32);

    // this evolves the game in the GPU
    gstart();
    for (int t = 1; t <= num_iterations / 2; t++) {
        //simple_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_2, d_game_1, gamedim);
        //simple_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_1, d_game_2, gamedim);

        //textures1D_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_2, 1, gamedim);
        //textures1D_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_1, 2, gamedim);

        //textures2D_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_2, 1, gamedim);
        //textures2D_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_1, 2, gamedim);

        /*
        textures2D_using_array_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_1, 1, gamedim);
        cudaMemcpyToArray(cuArray_game_1,0,0,d_game_1,sizeof(int)*gamesize,cudaMemcpyDeviceToDevice);
        textures2D_using_array_play_game_gpu<<<    numblocks  ,  numthreadsperblock   >>>(d_game_1, 1, gamedim);
        cudaMemcpyToArray(cuArray_game_1,0,0,d_game_1,sizeof(int)*gamesize,cudaMemcpyDeviceToDevice);
        */

        textures2D_using_array_and_surface_play_game_gpu<<<numblocks, numthreadsperblock>>>(1, gamedim);
        textures2D_using_array_and_surface_play_game_gpu<<<numblocks, numthreadsperblock>>>(2, gamedim);

        cudaDeviceSychronize();
    }
    gend(&gpu_time);

    //cudaMemcpy(h_game_3,d_game_1,sizeof(int)*gamesize,cudaMemcpyDeviceToHost);
    cudaMemcpyFromArray(h_game_3, cuArray_game_1, 0, 0, sizeof(int) * gamesize, cudaMemcpyDeviceToHost);

    // this evolves the game in the CPU
    //printgame(h_game_1,gamedim);
    cstart();
    for (int t = 1; t <= num_iterations / 2; t++) {
        play_game_cpu(h_game_2, h_game_1, gamedim);
        //printgame(h_game_2,gamedim);
        play_game_cpu(h_game_1, h_game_2, gamedim);
        //printgame(h_game_1,gamedim);
    }
    cend(&cpu_time);
    // when we leave the last game configuration is in h_game_2

    //check the solutions
    long int error_flag = 0;
    for (int i = 0; i < gamesize; i++) {
        if (h_game_1[i] != h_game_3[i]) {
            error_flag = error_flag + 1;
        }
    }

    printf("Error flag %ld and the time of CPU is %f and GPU is %f\n", error_flag, cpu_time, gpu_time);

    gerror(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    //printgame(h_game_1,gamedim);
    //printgame(h_game_3,gamedim);

    cudaUnbindTexture(tex_game_1);
    cudaUnbindTexture(tex_game_2);
    cudaUnbindTexture(tex2D_game_1_for_cuarray_use);
    cudaUnbindTexture(tex2D_game_2_for_cuarray_use);

    cudaFreeArray(cuArray_game_1);
    cudaFreeArray(cuArray_game_2);
    cudaFree(d_game_1);
    cudaFree(d_game_2);
    free(h_game_1);
    free(h_game_2);
    free(h_game_3);

    return 0;
}
