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
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < VSIZE) {
       C.elements[i] = A.elements[i] + B.elements[i];
    } 
}

#endif // #ifndef _VECTORADD_KERNEL_H_
