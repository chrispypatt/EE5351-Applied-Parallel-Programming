/* Vector Addition: C = A + B.
 * Device code.
 */

#ifndef _VECTORADD_KERNEL_H_
#define _VECTORADD_KERNEL_H_

#include <stdio.h>
#include "vectoradd.h"

// Vector addition kernel thread specification
__global__ void VectorAddKernel(Vector A, Vector B, Vector C)
{
	//Add the two vectors
	// unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
    // unsigned idx = x + y;
    // if(x < A && y < B) {
    //    C[idx] = A[idx] + B[idx];
    // } 

}

#endif // #ifndef _VECTORADD_KERNEL_H_
