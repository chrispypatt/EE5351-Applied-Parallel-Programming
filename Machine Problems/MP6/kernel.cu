#include <stdio.h>
#define BLOCK_SIZE 128

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < dim){
        float dot = 0.0;
        int row_start = csrRowPtr[row];
        int row_end = csrRowPtr[row+1];
        //Each kernel will calculate the dot of one row
        for (int i = row_start; i < row_end; i++){
            int col_idx = csrColIdx[i];
            dot += csrData[i] * inVector[col_idx];
        }
        outVector[row] += dot;
    }
}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {
    //Threads in the same block have similar amount of work (# elements to process)
    //which reduces divergence 
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < dim){ //Each kernel will calculate the dot of one row
        float dot = 0.0;
        //Iterate through the jds data for the row. 
        //JDS is condensed so we just loop for # of Non-zeros in row
        for (int i = 0; i < jdsRowNNZ[row]; i++){
            //get JDS data's value * inVector's matching column value
            int jds_idx = jdsColStartIdx[i] + row;
            int col_idx = jdsColIdx[jds_idx];
            dot += jdsData[jds_idx] * inVector[col_idx];
        }
        //put our dot into the correct output vector row.
        //jdsRowPerm holds the original rows of the permuted data
        outVector[jdsRowPerm[row]] += dot;
    }
}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {

    dim3 dimGrid(ceil(float(dim)/float(BLOCK_SIZE)), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    spmv_csr_kernel<<<dimGrid,dimBlock>>>(dim, csrRowPtr, csrColIdx, 
        csrData, inVector, outVector);
    cudaDeviceSynchronize(); 
}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {

    dim3 dimGrid(ceil(float(dim)/float(BLOCK_SIZE)), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    spmv_jds_kernel<<<dimGrid, dimBlock>>>(dim, jdsRowPerm, jdsRowNNZ, jdsColStartIdx,
        jdsColIdx, jdsData, inVector, outVector);
    cudaDeviceSynchronize(); 


}






