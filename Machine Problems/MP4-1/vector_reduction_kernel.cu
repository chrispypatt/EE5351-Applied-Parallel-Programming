#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *g_data, int n)
{
	//create shared memory for this block's reduction
	__shared__ unsigned int partialSum[2*BLOCK_SIZE];

	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockDim.x*blockIdx.x;

	//Load first element for this thread
	//check if we're within the input bounds
	if ((start+t) < n){ 
		partialSum[t] = g_data[start+t];
	}else{//If not in bounds, we want to add 0 to sum
		partialSum[t] = 0;
	}
	//Load second element for this thread
	//check if we're within the input bounds
	if ((start+ blockDim.x+t) < n){
		partialSum[blockDim.x+t] = g_data[start+blockDim.x+t];
	}else{//If not in bounds, we want to add 0 to sum
		partialSum[blockDim.x+t] = 0;
	}

	//loop and reduce block down to one value
	for (unsigned int stride = blockDim.x; stride >=1; stride >>=1){
		__syncthreads();
		if (t<stride){
			partialSum[t] += partialSum[t+stride];
		}
	}

	//g_data reduced to size g_data.size()/BLOCKSIZE
	//each block reduced to one value. put that value at block # we're in
	if (t==0){
		g_data[blockIdx.x] = partialSum[0];
	}
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
