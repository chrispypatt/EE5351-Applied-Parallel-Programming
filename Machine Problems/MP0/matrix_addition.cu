#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

const unsigned matrix_x = 4096;
const unsigned matrix_y = 4096;
const unsigned matrix_size = matrix_x * matrix_y;

__global__ void matrix_matrix_addition(float *C, const float * __restrict__ A, const float * __restrict__ B,
        const unsigned matrix_x, const unsigned matrix_y)
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned idx = x + y * matrix_x;
    if(x < matrix_x && y < matrix_y) {
       C[idx] = A[idx] + B[idx];
    } 
}
    

int main(int argc, char *argv[])
{
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    int i;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;
    float error = 0.0f;

    /* Allocate host memory */
    A_h = (float *)malloc(matrix_size * sizeof(float));
    if(A_h == NULL) FATAL("Unable to allocate host");
    B_h = (float *)malloc(matrix_size * sizeof(float));
    if(B_h == NULL) FATAL("Unable to allocate host");
    C_h = (float *)malloc(matrix_size * sizeof(float));
    if(C_h == NULL) FATAL("Unable to allocate host");


    /* Allocate device memory */
    cuda_ret = cudaMalloc((void **)&A_d, matrix_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void **)&B_d, matrix_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void **)&C_d, matrix_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");


    /* Initialize input matrixes */
    srand(time(NULL));
    for(i = 0; i < matrix_size; i++) {
        A_h[i] = 1.0 * rand() / RAND_MAX;
        B_h[i] = -1.0 * A_h[i];
    }
    cuda_ret = cudaMemcpy(A_d, A_h, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(B_d, B_h, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    dim_block.x = dim_block.y = 16; dim_block.z = 1;
    dim_grid.x = matrix_x / dim_block.x;
    if(matrix_x % dim_block.x != 0) dim_grid.x++;
    dim_grid.y = matrix_y / dim_block.y;
    if(matrix_y % dim_block.y != 0) dim_grid.y++;
    dim_grid.z = 1;
    matrix_matrix_addition<<<dim_grid, dim_block>>>(C_d, A_d, B_d, matrix_x, matrix_y);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
    

    /* Copy the result back */
    cuda_ret = cudaMemcpy(C_h, C_d, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    /* Check the results */
    for(i = 0; i < matrix_size; i++) error += C_h[i];
    if(error != 0.0) fprintf(stdout, "Error!\n");
    else fprintf(stdout, "Ok!\n");

    /* Free the mallocs */
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    free(A_h); free(B_h); free(C_h);

    return 0;
}
