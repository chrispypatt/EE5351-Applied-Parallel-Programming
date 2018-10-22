#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 1024
// You can use any other block size you wish.
#define BLOCK_SIZE 256

#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

unsigned int** scanBlockSums;
unsigned int** blockSums;
unsigned int* blockCounts;
unsigned int numAllocations;

// Host Helper Functions (allocate your own data structure...)

void preallocBlockSums(int num_elements){
	int blockCount = num_elements;
    unsigned int count = 0;
    
    while(blockCount > 1){
		blockCount = (unsigned int)ceil(float(blockCount)/float(2.f*BLOCK_SIZE));
		if(blockCount >= 1){
			count++;
		}
	}
	
	scanBlockSums = (unsigned int**)malloc(count * sizeof(unsigned int*));
	blockSums = (unsigned int**)malloc(count * sizeof(unsigned int*));
	blockCounts = (unsigned int*)malloc(count * sizeof(unsigned int));
	numAllocations = count;

    blockCount = num_elements;
    count = 0;
    while(blockCount > 1){
		blockCount = (unsigned int)ceil(float(blockCount)/float(2.f*BLOCK_SIZE));
		if(blockCount >= 1){
			cudaMalloc((void**)&(scanBlockSums[count]), blockCount * sizeof(unsigned int));
			cudaMalloc((void**)&(blockSums[count]), blockCount * sizeof(unsigned int));
			blockCounts[count] = blockCount;
			count++;
		}
    }
}

void deallocBlockSums(){
	for (int i = 0; i < numAllocations; i++){
		cudaFree(scanBlockSums[i]);
		cudaFree(blockSums[i]);
	}
	free(blockCounts);
	numAllocations = 0;
}

// Device Functions



// Kernel Functions
__global__ void prescan(unsigned int *g_odata,unsigned int *g_idata, unsigned int *S, int n){
	__shared__ unsigned int temp[2*BLOCK_SIZE];
	int t_idx = threadIdx.x, b_idx =  blockIdx.x;
	int idx = t_idx + b_idx * blockDim.x;
	int offset = 1;

	//Load into shared memory
	if (2*idx < n){
		temp[2*t_idx] = g_idata[2*idx]; 
	}else{
		temp[2*t_idx] = 0;
	}
	if (2*idx+1 < n){
		temp[2*t_idx+1] = g_idata[2*idx+1]; 
	}else{
		temp[2*t_idx+1] = 0;
	}
	for (int d = BLOCK_SIZE;d > 0; d>>=1){
		__syncthreads();
		
		if (t_idx < d){
			int ai = offset * (2*t_idx+1) - 1;
			int bi = offset * (2*t_idx+2) - 1;

			temp[bi] += temp[ai];
		}
		offset *=2;
	}

	if (t_idx == 0) { 
		S[b_idx] = temp[2*blockDim.x-1];
		temp[2*blockDim.x-1] = 0;
	}

	for (int d = 1; d < BLOCK_SIZE*2; d*=2){
		offset >>= 1;
		__syncthreads();

		if (t_idx < d){
			int ai = offset * (2*t_idx+1) - 1;
			int bi = offset * (2*t_idx+2) - 1;


			unsigned int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;	
		}
	}
	__syncthreads();
	if (2*idx <n){
		g_odata[2*idx] = temp[2*t_idx];  
	}
	if((2*idx + 1) < n){
		g_odata[2*idx+1] = temp[2*t_idx+1];
	} 
}

__global__ void addScannedBlockSum(unsigned int *g_odata, unsigned int *S, int n){//Sum up block sums to later blocks
	int t_idx = threadIdx.x, b_idx =  blockIdx.x;
	int idx = t_idx + b_idx * blockDim.x;
	if (2*idx <n){
		g_odata[2*idx] += S[b_idx]; 
	}
	if((2*idx + 1) < n){
		g_odata[2*idx+1] += S[b_idx];
	} 
}


// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{	
	int blocks = numElements/(2*BLOCK_SIZE);
	dim3 dimGrid(blocks, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	//Fill out array with exclusive scan on blocks. blockSums will contain each block's
	//final sum number for later use (only if applicable i.e. more than 1 kernel block)
	prescan<<<dimGrid,dimBlock>>>(outArray,inArray,blockSums[0],numElements);

	dim3 dimGridSums(dimGrid.x, 1, 1);

	for(int i = 0; i < numAllocations-1; i++){//Drill down until only one block in grid
		dimGridSums.x = (int)ceil(double(dimGridSums.x)/double(2*BLOCK_SIZE));
		//scan block sums
		prescan<<<dimGridSums, dimBlock>>>(scanBlockSums[i],blockSums[i],blockSums[i+1],blockCounts[i]); 
	}
	for(int j = numAllocations-2; j >= 0; j--){//need to propagate blockSums to later blocks (drill back up)
		dim3 dimGridSums(blockCounts[j+1], 1, 1);
		//Add Scanned Block Sum i to All Values of Scanned Block i + 1
		addScannedBlockSum<<<dimGridSums, dimBlock>>>(scanBlockSums[j],scanBlockSums[j+1],blockCounts[j]);
	}
	//finally add last block sum to our output
	addScannedBlockSum<<<dimGrid,dimBlock>>>(outArray,scanBlockSums[0],numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
