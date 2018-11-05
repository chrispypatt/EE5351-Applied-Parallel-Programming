#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    // Input dimension parameter
    unsigned int dim;
    enum Mode {CSR = 1, JDS};
    Mode mode;

    if (argc == 2) {
        mode = (Mode) atoi(argv[1]);
        dim = 1000;
    } else if (argc == 3) {
        mode = (Mode) atoi(argv[1]);
        dim = atoi(argv[2]);
    }
    if(argc < 2 || argc > 3 || (mode != CSR && mode != JDS)) {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./spmv <m>       # Mode: m, Matrix 1000 x 1000"
      "\n           ./spmv <m> <n>   # Mode: m, Matrix is n x n"
      "\n"
      "\n    Modes: 1 = CSR"
      "\n           2 = JDS"
      "\n\n");
        exit(0);
    }

    // CSR Matrix
    unsigned int *csrRowPtr, *csrColIdx;
    unsigned int *csrRowPtr_d, *csrColIdx_d;
    float *csrData;
    float *csrData_d;
    generateCSRMatrix(dim, &csrRowPtr, &csrColIdx, &csrData);

    // JDS Matrix
    unsigned int *jdsRowPerm, *jdsRowNNZ, *jdsColStartIdx, *jdsColIdx;
    unsigned int *jdsRowPerm_d, *jdsRowNNZ_d, *jdsColStartIdx_d, *jdsColIdx_d;
    float *jdsData;
    float *jdsData_d;
    if(mode == JDS) {
        csr2jds(dim, csrRowPtr, csrColIdx, csrData, &jdsRowPerm, &jdsRowNNZ, 
                &jdsColStartIdx, &jdsColIdx, &jdsData);
    }

    // Vectors
    float *inVector, *outVector;
    float *inVector_d, *outVector_d;

    inVector = (float*) malloc( sizeof(float)*dim );
    for (unsigned int i=0; i < dim; ++i) { inVector[i] = (rand()%100)/100.00; }

    outVector = (float*) malloc( sizeof(float)*dim );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Matrix dimensions: %u x %u\n"
           "    Number of non-zeros: %u\n", 
           dim, dim, csrRowPtr[dim]);
    if(mode == CSR) {
        printf("    Matrix storage format: CSR\n");
    } else if(mode == JDS) {
        printf("    Matrix storage format: JDS\n");
    }

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    // Allocate matrix
    const unsigned int NNZ = csrRowPtr[dim];
    if(mode == CSR) {

        cuda_ret = cudaMalloc((void**) &csrRowPtr_d, sizeof(unsigned int)*(dim + 1));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**) &csrColIdx_d, sizeof(unsigned int)*NNZ);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**) &csrData_d, sizeof(float)*NNZ);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    } else if(mode == JDS) {

        unsigned int maxRowNNZ = jdsRowNNZ[0]; // Largest number of non-zeros per row

        cuda_ret = cudaMalloc((void**) &jdsRowPerm_d,  sizeof(unsigned int)*dim);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**) &jdsRowNNZ_d,  sizeof(unsigned int)*dim);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**) &jdsColStartIdx_d,  sizeof(unsigned int)*maxRowNNZ);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**) &jdsColIdx_d,  sizeof(unsigned int)*NNZ);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**) &jdsData_d,  sizeof(float)*NNZ);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    }

    // Allocate vectors

    cuda_ret = cudaMalloc((void**) &inVector_d, sizeof(float)*dim);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cuda_ret = cudaMalloc((void**) &outVector_d, sizeof(float)*dim);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    // Copy matrix
    if(mode == CSR) {

        cuda_ret = cudaMemcpy(csrRowPtr_d, csrRowPtr, sizeof(unsigned int)*(dim + 1), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

        cuda_ret = cudaMemcpy(csrColIdx_d, csrColIdx, sizeof(unsigned int)*NNZ, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

        cuda_ret = cudaMemcpy(csrData_d, csrData, sizeof(float)*NNZ, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    } else if(mode == JDS) {

        unsigned int maxRowNNZ = jdsRowNNZ[0]; // Largest number of non-zeros per row

        cuda_ret = cudaMemcpy(jdsRowPerm_d, jdsRowPerm, sizeof(unsigned int)*dim, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

        cuda_ret = cudaMemcpy(jdsRowNNZ_d, jdsRowNNZ, sizeof(unsigned int)*dim, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

        cuda_ret = cudaMemcpy(jdsColStartIdx_d, jdsColStartIdx, sizeof(unsigned int)*maxRowNNZ, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

        cuda_ret = cudaMemcpy(jdsColIdx_d, jdsColIdx, sizeof(unsigned int)*NNZ, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

        cuda_ret = cudaMemcpy(jdsData_d, jdsData, sizeof(float)*NNZ, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    }

    // Copy vectors

    cuda_ret = cudaMemcpy(inVector_d, inVector, sizeof(float)*dim, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    if(mode == CSR) {
        spmv_csr(dim, csrRowPtr_d, csrColIdx_d, csrData_d, inVector_d, 
            outVector_d);
    } else if(mode == JDS) {
        spmv_jds(dim, jdsRowPerm_d, jdsRowNNZ_d, jdsColStartIdx_d, jdsColIdx_d, 
            jdsData_d, inVector_d, outVector_d);
    }

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(outVector, outVector_d, sizeof(float)*dim, 
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(dim, csrRowPtr, csrColIdx, csrData, inVector, outVector);

    // Free memory ------------------------------------------------------------

    free(csrRowPtr);
    free(csrColIdx);
    free(csrData);
    if(mode == JDS) {
        free(jdsRowPerm);
        free(jdsRowNNZ);
        free(jdsColStartIdx);
        free(jdsColIdx);
        free(jdsData);
    }
    free(inVector);
    free(outVector);

    if(mode == CSR) {
        cudaFree(csrRowPtr_d);
        cudaFree(csrColIdx_d);
        cudaFree(csrData_d);
    } else if (mode == JDS) {
        cudaFree(jdsRowPerm_d);
        cudaFree(jdsRowNNZ_d);
        cudaFree(jdsColStartIdx_d);
        cudaFree(jdsColIdx_d);
        cudaFree(jdsData_d);
    }
    cudaFree(inVector_d);
    cudaFree(outVector_d);

    return 0;

}

