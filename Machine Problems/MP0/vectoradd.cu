/* Vector addition: C = A + B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//#include "cutil.h"

// includes, kernels
#include "vectoradd_kernel.cu"

#define MAXLINE 100000

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int);

Vector AllocateDeviceVector(const Vector V);
Vector AllocateVector(int size, int init);
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost);
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice);
int ReadFile(Vector* V, char* file_name);
void WriteFile(Vector V, char* file_name);

void VectorAddOnDevice(const Vector A, const Vector B, Vector C);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Vectors for the program
	Vector A;
	Vector B;
	Vector C;
	// Number of elements in the vectors
	unsigned int size_elements = VSIZE;
	int errorA = 0, errorB = 0;
	
	srand(2012);
	
	// Check command line for input vector files
	if(argc != 3 && argc != 4) 
	{
		// No inputs provided
		// Allocate and initialize the vectors
		A  = AllocateVector(VSIZE, 1);
		B  = AllocateVector(VSIZE, 1);
		C  = AllocateVector(VSIZE, 0);
	}
	else
	{
		// Inputs provided
		// Allocate and read source vectors from disk
		A  = AllocateVector(VSIZE, 0);
		B  = AllocateVector(VSIZE, 0);		
		C  = AllocateVector(VSIZE, 0);
		errorA = ReadFile(&A, argv[1]);
		errorB = ReadFile(&B, argv[2]);
		// check for read errors
		if(errorA != size_elements || errorB != size_elements)
		{
			printf("Error reading input files %d, %d\n", errorA, errorB);
			return 1;
		}
	}

	// A + B on the device
    VectorAddOnDevice(A, B, C);
    
    // compute the vector addition on the CPU for comparison
    Vector reference = AllocateVector(VSIZE, 0);
    computeGold(reference.elements, A.elements, B.elements, VSIZE);
        
    // check if the device result is equivalent to the expected solution
    //CUTBoolean res = cutComparefe(reference.elements, C.elements, 
	//								size_elements, 0.0001f);
    unsigned res = 1;
    for (unsigned i = 0; i < size_elements; i++)
        if (abs(reference.elements[i] - C.elements[i]) > 0.0001f)
            res = 0;

    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    
    // output result if output file is requested
    if(argc == 4)
    {
		WriteFile(C, argv[3]);
	}
	else if(argc == 2)
	{
	    WriteFile(C, argv[1]);
	}    

	// Free host matrices
    free(A.elements);
    A.elements = NULL;
    free(B.elements);
    B.elements = NULL;
    free(C.elements);
    C.elements = NULL;
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void VectorAddOnDevice(const Vector A, const Vector B, Vector C)
{
	//Interface host call to the device kernel code and invoke the kernel

    //* steps:
    //* 1. allocate device vectors d_A, d_B and d_C with length same as input vectors
    Vector d_A  = AllocateDeviceVector(A);
    Vector d_B  = AllocateDeviceVector(B);
    Vector d_C  = AllocateDeviceVector(C);

    //* 2. copy A to d_A, B to d_B
    CopyToDeviceVector(d_A,A);
    CopyToDeviceVector(d_B,B);
    CopyToDeviceVector(d_C,C);

    //* 3. launch kernel to compute d_C = d_A + d_B
    VectorAddKernel(d_A,d_B,d_C);

    //* 4. copy d_C back to host vector C
    CopyFromDeviceVector(C,d_C);

    //* 5. free device vectors d_A, d_B, d_C
    free(d_A.elements);
    d_A.elements = NULL;
    free(d_B.elements);
    d_B.elements = NULL;
    free(d_C.elements);
    d_C.elements = NULL;
}

// Allocate a device vector of same size as V.
Vector AllocateDeviceVector(const Vector V)
{
    Vector Vdevice = V;
    int size = V.length * sizeof(float);
    cudaError_t cuda_ret = cudaMalloc((void**)&Vdevice.elements, size);
    if(cuda_ret != cudaSuccess) {
        printf("Unable to allocate device memory");
        exit(0);
    }
    return Vdevice;
}

// Allocate a vector of dimensions length
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Vector AllocateVector(int length, int init)
{
    Vector V;
    V.length = length;
    V.elements = NULL;
		
	V.elements = (float*) malloc(length*sizeof(float));

	for(unsigned int i = 0; i < V.length; i++)
	{
		V.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	}
    return V;
}	

// Copy a host vector to a device vector.
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost)
{
    int size = Vhost.length * sizeof(float);
    Vdevice.length = Vhost.length;
    cudaMemcpy(Vdevice.elements, Vhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device vector to a host vector.
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice)
{
    int size = Vdevice.length * sizeof(float);
    cudaMemcpy(Vhost.elements, Vdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Read a floating point vector in from file
int ReadFile(Vector* V, char* file_name)
{
	unsigned int data_read = VSIZE;
	FILE* input = fopen(file_name, "r");
    char vector_string[MAXLINE];
    fgets(vector_string, MAXLINE, input);
    char* part = strtok(vector_string, " ");
    for (unsigned i = 0; i < VSIZE; i++) {
        V->elements[i] = atof(part);
        part = strtok(NULL, " ");
    }
	return data_read;
}

// Write a floating point vector to file
void WriteFile(Vector V, char* file_name)
{
    FILE* output = fopen(file_name, "w");
    for (unsigned i = 0; i < VSIZE; i++) {
        fprintf(output, "%f ", V.elements[i]);
    }
}
