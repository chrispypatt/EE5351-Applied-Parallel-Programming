#include <stdio.h>

enum reduction_op { max, min, sum, count, mean };

template<typename T>
struct Groupby
{
    T* key_columns[];
	int* key_ends;
	int* key_counts;
	int num_key_columns;
	int num_key_rows;

	T* value_columns[];
	int* output_index[];
	int num_value_columns;
	int num_value_rows;

	reduction_op ops[];
	int num_ops;

	T* output_keys[];
	T* output_values[];
	int num_output_rows;

};



//max, min, sum, count, and arithmetic mean

__global__ void max_kernel(){
	//create shared memory for this block's reduction
	__shared__ T s_col_data[BLOCK_SIZE];
	__shared__ T s_sums[data.num_unique_keys];
	__shared__ T s_counts[data.num_unique_keys];

	unsigned int t = threadIdx.x;
	unsigned int data_column_start = (col+data.num_key_columns)*input.num_value_rows;//get us into the column we are trying to access
	unsigned int output_column_start = col*data.num_output_rows;

	//Load first element for this thread
	//check if we're within the height bounds for this column
	if (t < num_rows){ 
		s_col_data[t] = data.value_columns[t+data_column_start];
	}else{//If not in bounds, we want to add 0 to sum
		s_col_data[t] = 0;
	}

	atomicMax(&(s_sums[data.output_index[t]]), s_col_data[t]);
	__syncthreads();

	if(t<data.num_output_rows){
		atomicMax(&(data.output_values[t+output_column_start]), s_sums[t]);
	}
}

__global__ void min_kernel(){
	//create shared memory for this block's reduction
	__shared__ T s_col_data[BLOCK_SIZE];
	__shared__ T s_sums[data.num_unique_keys];
	__shared__ T s_counts[data.num_unique_keys];

	unsigned int t = threadIdx.x;
	unsigned int data_column_start = (col+data.num_key_columns)*input.num_value_rows;//get us into the column we are trying to access
	unsigned int output_column_start = col*data.num_output_rows;

	//Load first element for this thread
	//check if we're within the height bounds for this column
	if (t < num_rows){ 
		s_col_data[t] = data.value_columns[t+data_column_start];
	}else{//If not in bounds, we want to add 0 to sum
		s_col_data[t] = 0;
	}

	atomicMin(&(s_sums[data.output_index[t]]), s_col_data[t]);
	__syncthreads();

	if(t<data.num_output_rows){
		atomicMin(&(data.output_values[t+output_column_start]), s_sums[t]);
	}
}

template <typename T>
__global__ void sum_kernel(Groupby data, int col){
	//create shared memory for this block's reduction
	__shared__ T s_col_data[BLOCK_SIZE];
	__shared__ T s_sums[data.num_unique_keys];
	__shared__ T s_counts[data.num_unique_keys];

	unsigned int t = threadIdx.x;
	unsigned int data_column_start = (col+data.num_key_columns)*input.num_value_rows;//get us into the column we are trying to access
	unsigned int output_column_start = col*data.num_output_rows;

	//Load first element for this thread
	//check if we're within the height bounds for this column
	if (t < num_rows){ 
		s_col_data[t] = data.value_columns[t+data_column_start];
	}else{//If not in bounds, we want to add 0 to sum
		s_col_data[t] = 0;
	}

	atomicAdd(&(s_sums[data.output_index[t]]), s_col_data[t]);
	atomicAdd(&(s_counts[data.output_index[t]]), 1);
	__syncthreads();

	if(t<data.num_output_rows){
		atomicAdd(&(data.output_values[t+output_column_start]), s_sums[t]);
		atomicAdd(&(data.key_counts[t+output_column_start]), s_counts[t]);
	}
}

template <typename T>
__global__ void count_kernel(){
	//create shared memory for this block's reduction
	__shared__ T s_counts[data.num_unique_keys];

	unsigned int t = threadIdx.x;
	unsigned int output_column_start = col*data.num_output_rows;


	atomicAdd(&(s_counts[data.output_index[t]]), 1);
	__syncthreads();

	if(t<data.num_output_rows){
		atomicAdd(&(data.key_counts[t+output_column_start]), s_counts[t]);
	}
}

template <typename T>
__global__ void arithmetic_mean_kernel(Groupby data){
	unsigned int t = threadIdx.x;
	unsigned int output_column_start = col*data.num_output_rows;

	if (t<data.num_output_rows){
		data.output_values[t+output_column_start] = data.output_values[t+output_column_start]/data.key_counts;
	}
}

//Launch reduction kernels for each column based on their specified operation
void perform_operators(Groupby input, Groupby output)
{
	//set kernel size
	dim3 dimGrid(ceil(double(input.num_key_rows)/double(BLOCK_SIZE)),1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);

	for (int i = 0; i<input.num_ops, i++){
		switch(input.ops[i]){
			case max:
				max_kernel<<<dimGrid,dimBlock>>>(data);	
				break;
			case min:
				min_kernel<<<dimGrid,dimBlock>>>(data);	
				break;
			case sum:
				sum_kernel<<<dimGrid,dimBlock>>>(data);
				break;
			case count:
				count_kernel<<<dimGrid,dimBlock>>>(data);	
				break;
			case mean:
				//we can call sum kernel and then make sure to perform a division after
				sum_kernel<<<dimGrid,dimBlock>>>(data);
				dim3 dimGrid1(ceil(double(input.num_unique_keys)/double(BLOCK_SIZE)),1,1);
				dim3 dimBlock1(BLOCK_SIZE,1,1);
				arithmetic_mean_kernel<<<dimGrid1,dimBlock1>>>(data);
				//now we have the mean
				break;
		}
		cudaDeviceSynchronize();
	}


	// copyFromDeviceArray(h_data,d_data,1);
	// cudaFree(&d_data);
	return h_data[0]; //Input array now reduced to sum of all elements
}