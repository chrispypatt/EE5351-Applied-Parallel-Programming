#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void generateCSRMatrix(unsigned int dim, unsigned int **csrRowPtr,
    unsigned int **csrColIdx, float **csrData) {

    const unsigned int MAX_NNZ_PER_ROW = dim/10 + 1;

    *csrRowPtr = (unsigned int*) malloc( sizeof(unsigned int)*(dim + 1) );
    (*csrRowPtr)[0] = 0;
    for(unsigned int rowIdx = 0; rowIdx < dim; ++rowIdx) {
        unsigned int rowNNZ = rand()%(MAX_NNZ_PER_ROW + 1);
        (*csrRowPtr)[rowIdx + 1] = (*csrRowPtr)[rowIdx] + rowNNZ;
    }

    const unsigned int NNZ = (*csrRowPtr)[dim];
    *csrColIdx = (unsigned int*) malloc( sizeof(unsigned int)*NNZ );
    *csrData = (float*) malloc( sizeof(float)*NNZ );
    for (unsigned int i = 0; i < NNZ; ++i) {
        (*csrColIdx)[i] = rand()%dim; 
        (*csrData)[i] = (rand()%100)/100.00; 
    }

}

void quicksort(unsigned int *data, unsigned int *key, unsigned int start, unsigned int end) {
    if((end - start + 1) > 1) {
        unsigned int left = start, right = end;
        unsigned int pivot = key[right];
        while(left <= right) {
            while(key[left] > pivot) {
                left = left + 1;
            }
            while(key[right] < pivot) {
                right = right - 1;
            }
            if(left <= right) {
                unsigned int tmp =  key[left];  key[left] =  key[right];  key[right] = tmp;
                             tmp = data[left]; data[left] = data[right]; data[right] = tmp;
                left = left + 1;
                right = right - 1;
            }
        }
        quicksort(data, key, start, right);
        quicksort(data, key, left, end);
    }
}

void csr2jds(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx,
    float *csrData, unsigned int **jdsRowPerm, unsigned int **jdsRowNNZ,
    unsigned int **jdsColStartIdx, unsigned int **jdsColIdx, float **jdsData) {

    // Row Permutation Vector
    *jdsRowPerm = (unsigned int*) malloc( sizeof(unsigned int)*dim );
    for(unsigned int rowIdx = 0; rowIdx < dim; ++rowIdx) {
        (*jdsRowPerm)[rowIdx] = rowIdx;
    }

    // Number of non-zeros per row
    *jdsRowNNZ = (unsigned int*) malloc( sizeof(unsigned int)*dim );
    for(unsigned int rowIdx = 0; rowIdx < dim; ++rowIdx) {
        (*jdsRowNNZ)[rowIdx] = csrRowPtr[rowIdx + 1] - csrRowPtr[rowIdx];
    }

    // Sort rows by number of non-zeros
    quicksort(*jdsRowPerm, *jdsRowNNZ, 0, dim - 1);

    // Starting point of each compressed column
    unsigned int maxRowNNZ = (*jdsRowNNZ)[0]; // Largest number of non-zeros per row
    *jdsColStartIdx = (unsigned int*) malloc( sizeof(unsigned int)*maxRowNNZ );
    (*jdsColStartIdx)[0] = 0; // First column starts at 0
    for(unsigned int col = 0; col < maxRowNNZ - 1; ++col) {
        // Count the number of rows with entries in this column
        unsigned int count = 0;
        for(unsigned int idx = 0; idx < dim; ++idx) {
            if((*jdsRowNNZ)[idx] > col) {
                ++count;
            }
        }
        (*jdsColStartIdx)[col + 1] = (*jdsColStartIdx)[col] + count;
    }

    // Sort the column indexes and data
    const unsigned int NNZ = csrRowPtr[dim];
    *jdsColIdx = (unsigned int*) malloc( sizeof(unsigned int)*NNZ );
    *jdsData = (float*) malloc( sizeof(float)*NNZ );
    for(unsigned int idx = 0; idx < dim; ++idx) { // For every row
        unsigned int row = (*jdsRowPerm)[idx];
        unsigned int rowNNZ = (*jdsRowNNZ)[idx];
        for(unsigned int nnzIdx = 0; nnzIdx < rowNNZ; ++nnzIdx) {
            unsigned int jdsPos = (*jdsColStartIdx)[nnzIdx] + idx;
            unsigned int csrPos = csrRowPtr[row] + nnzIdx;
            (*jdsColIdx)[jdsPos] = csrColIdx[csrPos];
            (*jdsData)[jdsPos] = csrData[csrPos];
        }
    }
}


void verify(unsigned int dim, unsigned int *csrRowPtr, unsigned int *csrColIdx, 
    float* csrData, float* inVector, float *outVector) {

    const float relativeTolerance = 1e-6;

    for(int row = 0; row < dim; ++row) {
        float result = 0.0f;
        unsigned int start = csrRowPtr[row];
        unsigned int end = csrRowPtr[row + 1];
        for(int elemIdx = start; elemIdx < end; ++elemIdx) {
            unsigned int colIdx = csrColIdx[elemIdx];
            result += csrData[elemIdx]*inVector[colIdx];
        }
        float relativeError = (result - outVector[row])/result;
        if (relativeError > relativeTolerance
                || relativeError < -relativeTolerance) {
            printf("TEST FAILED at row %d: CPU = %f, GPU = %f\n\n", 
                row, result, outVector[row]);
            exit(0);
        }
    }
    printf("TEST PASSED\n\n");

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

