/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    double TILE_WIDTH = 16.0;

    __shared__ float shared_M[16][16];
    __shared__ float shared_N[16][16];

    //Get block and thread idxs to load in tiles
    int by = blockIdx.y, bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;

    int p_row = by*TILE_WIDTH+ty;
    int p_col = bx*TILE_WIDTH+tx;
    float Pvalue = 0;
    
    for (int m = 0; m < ceil(double(M.width)/TILE_WIDTH); ++m){
        int M_col = m * TILE_WIDTH + tx;
        int N_row = m * TILE_WIDTH + ty;
        //Each thread fill up piece of shared memory.
        //If out of bounds, pad with 0. This means our tile did not fit matrix perfectly
        if (M_col < M.width && p_row < M.height){
            shared_M[ty][tx] = M.elements[p_row * M.width + M_col];
        }else{
            shared_M[ty][tx] = 0;
        }
        if (N_row < N.height && p_col < N.width){
            shared_N[ty][tx] = N.elements[N_row * N.width + p_col];
        }else{
            shared_N[ty][tx] = 0;
        } 
        //synchronize here so we know all the data is loaded into SM
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k){
            //Perform calculation with what we now have in shared memory
            Pvalue += shared_M[ty][k] * shared_N[k][tx];
            __syncthreads();
        }
    }
    //Only if thread is in bounds of output matrix, save calculated value
    if (p_row < P.height && p_col < P.width){
        P.elements[p_row*P.width+p_col] = Pvalue;
    }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
