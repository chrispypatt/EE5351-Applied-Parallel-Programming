/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>

// includes, kernels
#include "matrixmul_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);
bool CompareMatrices(Matrix A, Matrix B);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    // Matrices for the program
    Matrix  M;
    Matrix  N;
    Matrix  P;
    // Number of elements in the solution matrix
    //  Assuming square matrices, so the sizes of M, N and P are equal
    unsigned int size_elements = WP * HP;
    int errorM = 0, errorN = 0;

    srand(2012);

    // Check command line for input matrix files
    if(argc != 3 && argc != 4)
    {
        // No inputs provided
        // Allocate and initialize the matrices
        M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
        N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
        P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
    }
    else
    {
        // Inputs provided
        // Allocate and read source matrices from disk
        M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
        N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
        P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
        errorM = ReadFile(&M, argv[1]);
        errorN = ReadFile(&N, argv[2]);
        // check for read errors
        if(errorM != size_elements || errorN != size_elements)
        {
            printf("Error reading input files %d, %d\n", errorM, errorN);
            return 1;
        }
    }

    // M * N on the device
    MatrixMulOnDevice(M, N, P);

    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
    computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);

    // check if the device result is equivalent to the expected solution
    bool res = CompareMatrices(reference, P);
    printf("Test %s\n", res ? "PASSED" : "FAILED");

    // output result if output file is requested
    if(argc == 4)
    {
        WriteFile(P, argv[3]);
    }
    else if(argc == 2)
    {
        WriteFile(P, argv[1]);
    }

    // Free host matrices
    free(M.elements);
    M.elements = NULL;
    free(N.elements);
    N.elements = NULL;
    free(P.elements);
    P.elements = NULL;
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    //Interface host call to the device kernel code and invoke the kernel



}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.
//	If init == 1, perform random initialization.
Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    M.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < M.height * M.width; i++)
    {
        M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
    }
    return M;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,
               cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size,
               cudaMemcpyDeviceToHost);
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is
//  equals M.height * M.width, and 1 otherwise
int ReadFile(Matrix* M, char* file_name)
{
    unsigned int data_read = M->width * M->height;
    FILE* input = fopen(file_name, "r");
    unsigned i = 0;
    for (i = 0; i < data_read; i++)
        fscanf(input, "%f", &(M->elements[i]));
    return data_read;
}

// Write a floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
    unsigned int size = M.width * M.height;
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < size; i++)
        fprintf(output, "%f ", M.elements[i]);
}

// returns true iff A and B have same elements in same order
bool CompareMatrices(Matrix A, Matrix B) {
    unsigned int size = A.width * A.height;

    if ( (A.width != B.width) || (A.height != B.height) )
        return false;

    for (unsigned i = 0; i < size; i++)
        if (abs(A.elements[i] - B.elements[i]) > 0.0001f) 
            return false;
    return true;
}
