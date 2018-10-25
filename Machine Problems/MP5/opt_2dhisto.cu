#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

//abandoned, not possible - 1 hour while setting up functional non optimized kernel
// __device__ uint8_t atomicAdd8Int(uint8_t* address, uint8_t val){ //function to write to uint8_t atomically
// 	unsigned long long int* address_as_ull = (unsigned long long int*)address;
// 	unsigned long long int old = *address_as_ull, assumed;
// 	do{
// 		assumed = old;	// READ
// 		old = atomicCAS(address_as_ull, assumed,val + assumed);	// MODIFY + WRITE
// 	} while (assumed != old);
// 	return old;
// }

__global__ void histogram_kernel(uint32_t *buff, long size, int *histo){
	//start index into buffer
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	//stride is however many threads we have running 
	int stride = blockDim.x * gridDim.x; //memory coalescing

	__shared__ unsigned int histo_private[HISTO_WIDTH];
	if (threadIdx.x < HISTO_WIDTH) histo_private[threadIdx.x] = 0;
	__syncthreads();
	
	while(index < size){
		if (histo_private[buff[index]] < UINT8_MAXIMUM){ //Dont rollover!!
			atomicAdd(&(histo_private[buff[index]]), 1);
			index += stride;
		}
	}
	__syncthreads();
	if(threadIdx.x < HISTO_WIDTH){
		atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
	}
}

__global__ void convert_int2uint8(int *int_histo, int size, uint8_t *histo){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size){
		if (int_histo[index]>UINT8_MAXIMUM){
			histo[index] = UINT8_MAXIMUM;
		}else{
			histo[index] = (uint8_t)int_histo[index];
		}
	}
}

void opt_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH])
{
	long input_size = height*width;
	int histo_size = HISTO_HEIGHT*HISTO_WIDTH;
	//init bins to zero
	memset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(bins[0]));
	//setup our device histogram
	uint32_t* d_input;
	cudaMalloc((void**)&d_input, input_size*sizeof(uint32_t));
	for(int i = 0; i < height; ++i){
		//copy each row of input array to device 
		CopyInputToDeviceArray((d_input+i*width), width, input[i]);
    }	

	//uint8_t histogram for output
	uint8_t* d_histo;
	cudaMalloc((void**)&d_histo, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
	cudaMemset(d_histo, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

	//int histogram for calculation. atomicadd works with this
	int* d_int_histo;
	cudaMalloc((void**)&d_int_histo, HISTO_HEIGHT*HISTO_WIDTH*sizeof(int));
	cudaMemset(d_int_histo, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(int));


	dim3 dimGrid(ceil((float)input_size/(float)BLOCK_SIZE),1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);

	histogram_kernel<<<dimGrid,dimBlock>>>(d_input,input_size,d_int_histo);
    cudaDeviceSynchronize();

	dimGrid.x = (ceil((float)histo_size/(float)BLOCK_SIZE),1,1);

	convert_int2uint8<<<dimGrid,dimBlock>>>(d_int_histo,histo_size,d_histo);
    cudaDeviceSynchronize();

	CopyBinsFromDeviceArray(bins,HISTO_HEIGHT,HISTO_WIDTH,d_histo);

	cudaFree(d_histo);
	cudaFree(d_int_histo);
	cudaFree(d_input);
}

/* Include below the implementation of any other functions you need */

//copy data (input) from the host to our device
void CopyInputToDeviceArray(uint32_t* Adevice, size_t width, uint32_t* Ahost){
    int size = width * sizeof(uint32_t);
    cudaMemcpy(Adevice, Ahost, size, cudaMemcpyHostToDevice);
}

//copy data back from device to our host
void CopyBinsFromDeviceArray( uint8_t* Ahost, size_t height, size_t width,uint8_t* Adevice){
    int size = width * height * sizeof(uint8_t);
    cudaMemcpy(Ahost, Adevice, size, cudaMemcpyDeviceToHost);
}
