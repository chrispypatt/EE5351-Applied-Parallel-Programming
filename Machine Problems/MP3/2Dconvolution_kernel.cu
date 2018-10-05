#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// Matrix multiplication kernel thread specification
__global__ void ConvolutionKernel(Matrix N, Matrix P)
{
    __shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y, tx = threadIdx.x;

    //thread index mapping into output matrix P[row_P][col_P]
    int col_P = blockIdx.x * TILE_SIZE + tx;
    int row_P = blockIdx.y * TILE_SIZE + ty; 

    //thread indexing for loading N with ghosts 👻 and halos
    int col_N = col_P - KS_DIV_2;
    int row_N = row_P - KS_DIV_2;

    //Load tile from input
    if((col_N >= 0 && col_N < N.width) && (row_N >= 0 && row_N < N.height)){
        N_s[ty][tx] = N.elements[row_N*N.width+col_N];
    }else{//If outside of bounds, load ghost 👻 cell 
        N_s[ty][tx] = 0.0;
    }
    __syncthreads();

    if (tx < TILE_SIZE && ty < TILE_SIZE){
        float Pvalue = 0.0; 
        for (int a = 0; a < KERNEL_SIZE; a++) { //traverse filter by row (a = filter row)
            for (int b = 0; b < KERNEL_SIZE; b++) { //traverse each row by col (b = filter col)
                Pvalue += Mc[a*KERNEL_SIZE+b]*N_s[a+ty][b+tx];
            }
        } 
        if(col_P < P.width && row_P < P.height){ //If thread is within bounds of P output
            P.elements[row_P*P.width+col_P] = Pvalue; 
        }
    }
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
