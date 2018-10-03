#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// Matrix multiplication kernel thread specification
__global__ void ConvolutionKernel(Matrix N, Matrix P)
{
    __shared__ float N_s[TILE_SIZE][TILE_SIZE];

    int ty = threadIdx.y, tx = threadIdx.x;

    //This thread will compute P[Prow][Pcol]
    int Pcol = blockIdx.x * TILE_SIZE + tx;
    int Prow = blockIdx.y * TILE_SIZE + ty; 

    //Get the thread's row and col for loading N image
    //TODO: Double check this grabs correct elements from tile
    int Ncol = Pcol - KS_DIV_2;
    int Nrow = Prow - KS_DIV_2;

    //Load in this tile of N (set to zero if element outside matrix)
    if((Ncol > -1 && Ncol < N.width) && (Nrow > -1 && Nrow < N.height)){
        N_s[tx][ty] = N[Nrow*N.width+Ncol];
    }else{
        N_s[tx][ty] = 0.0;
    }
    __syncthreads();

    float Pvalue = 0.0; 
    for (int a = 0; a < KERNEL_SIZE; a++) { //traverse filter by row (a = filter row)
        for (int b = 0; b < KERNEL_SIZE; b++) { //traverse each row by col (b = filter col)
        //TODO: Finish this indexing for calculation
            Pvalue += Mc[a*KERNEL_SIZE+b]*N_s[row + a − 2][col + b − 2];
        }
    } 
    if(Pcol < P.width && Prow < P.height){ //If thread is within bounds of P output
        P.elements[Prow*KERNEL_SIZE+Pcol] = Pvalue; 
    }
}

#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
