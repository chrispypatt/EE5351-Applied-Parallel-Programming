#include <stdio.h>

__global__ void spmv_csr_kernel(unsigned int dim, unsigned int *csrRowPtr, 
    unsigned int *csrColIdx, float *csrData, float *inVector, 
    float *outVector) {

    // INSERT KERNEL CODE HERE

}

__global__ void spmv_jds_kernel(unsigned int dim, unsigned int *jdsRowPerm, 
    unsigned int *jdsRowNNZ, unsigned int *jdsColStartIdx, 
    unsigned int *jdsColIdx, float *jdsData, float* inVector,
    float *outVector) {

    // INSERT KERNEL CODE HERE

}

void spmv_csr(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float *csrData, float *inVector, float *outVector) {

    // INSERT CODE HERE

}

void spmv_jds(unsigned int dim, unsigned int *jdsRowPerm, unsigned int *jdsRowNNZ, 
    unsigned int *jdsColStartIdx, unsigned int *jdsColIdx, float *jdsData, 
    float* inVector, float *outVector) {

    // INSERT CODE HERE

}






