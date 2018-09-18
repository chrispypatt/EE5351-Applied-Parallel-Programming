/* Matrix multiplication: P = M * N.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  //Multiply the two matrices
  int Row = blockIdx.y*blockDim.y + threadIdx.y;
  int Col = blockIdx.x*blockDim.x + threadIdx.x;
  if ((Row < M.height) && (Col < M.width)){
    float Pvalue = 0.0;
    for(int k = 0; k < M.width; ++k)
      Pvalue += M.elements[Row*M.width+k] * N.elements[k*M.width+Col];
    P.elements[Row*M.width+Col] = Pvalue;
  }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
