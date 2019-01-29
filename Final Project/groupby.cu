#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/host_vector.h>

#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include "cpuGroupby.h"
#include "HashFunc.cuh"
#include <thrust/iterator/permutation_iterator.h>

#define BLOCK_SIZE 1024 // GTX 1080 only support 1024 thread per block

// original code can be seen at https://stackoverflow.com/questions/28607171/sort-2d-array-in-cuda-with-thrust
// modified for column major
template <typename T>
struct my_sort_functor
{
    int num_columns, num_rows;
    T* key_data; // 1D Array
    my_sort_functor(int __num_columns, int __num_rows, T* __key_data): num_columns(__num_columns), num_rows(__num_rows), key_data(__key_data) {};
    
    __host__ __device__
    bool operator()(const int idx1, const int idx2) const
    {
        bool flip = false;
        for (int i = 0; i < num_columns; ++i) {
            T data1 = key_data[i * num_rows + idx1];
            T data2 = key_data[i * num_rows + idx2];
            if (data1 > data2) break;
            else if (data1 < data2) {
                flip = true;
                break;
            }
        }
        return flip;
    }
};

// check current element and previous element, if not same set 1
template<typename T>
__global__ void identify_bound(T* key_columns, int num_key_rows, int num_key_columns,
	uint32_t* result_array)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int prev_idx = (tidx == 0) ? 0 : (tidx - 1);
	int result = 0;

	if (tidx < num_key_rows) {
		for (int i = 0; i < num_key_columns; ++i) {
			if (key_columns[i*num_key_rows+prev_idx] != key_columns[i*num_key_rows+tidx]) {
				result = 1;
				break;
			}
		}
		result_array[tidx] = result;
	}
}


//Launch reduction kernels for each column based on their specified operation
template <typename T>
void groupby_GPU(T* key_columns, int num_key_columns, int num_key_rows,
	T* value_columns, int num_value_columns, int num_value_rows,
	reductionType* ops, int num_ops, T* output_keys, T* output_values, int &num_output_rows)
{
	//Perform hashing
	uint32_t dimBlock = BLOCK_SIZE;
	uint32_t dimGrid = (num_key_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;

	typedef typename thrust::device_vector<T>::iterator ElementIterator;
	typedef thrust::device_vector<int>::iterator IndexIterator;

	// use device vector 


	thrust::device_vector<T> d_keys(key_columns, key_columns + num_key_rows * num_key_columns);
	T* d_keys_raw = thrust::raw_pointer_cast(d_keys.data());
	thrust::device_vector<T> d_sorted_keys = d_keys;
	T* d_sorted_keys_raw = thrust::raw_pointer_cast(d_sorted_keys.data());

	// create original index

	thrust::device_vector<int> d_i(num_key_rows);
	thrust::sequence(thrust::device, d_i.begin(), d_i.end()); 
	int * d_i_raw = thrust::raw_pointer_cast(d_i.data());

	// sort the index according to values in d_keys and distributed values to d_sorted_keys
	thrust::sort(d_i.begin(), d_i.end(), my_sort_functor<T>(num_key_columns, num_key_rows, d_keys_raw));
	
	for (int i = 0; i<num_key_columns; i++){//i represents column of key 
		thrust::device_vector<T> d_key_column(d_keys.begin() + (i*num_key_rows),d_keys.begin() + ((i+1)*num_key_rows));
		thrust::permutation_iterator<ElementIterator,IndexIterator> iter(d_key_column.begin(), d_i.begin());
		thrust::copy_n(iter, num_key_rows, d_sorted_keys.begin()+i*num_key_rows);
	}	

	uint32_t* hash_keys;
	cudaMalloc((void **) &hash_keys, num_key_rows * sizeof(uint32_t));
	thrust::device_ptr<uint32_t> d_hash_keys(hash_keys);
	thrust::fill(d_hash_keys, d_hash_keys + num_key_rows, (int) 0);

	// check the boundary then scan the boundary
	identify_bound<<<dimGrid, dimBlock>>>(d_sorted_keys_raw, num_key_rows, num_key_columns, hash_keys);
	thrust::inclusive_scan(thrust::device, d_hash_keys, d_hash_keys + num_key_rows, d_hash_keys);

	// Now the keys in d_sorted_keys should be sorted and d_hash_keys will have identical value for identical keys, note the value is already sorted
	// so can run reduce_by_key directly on the sorted keys to get unique keys

	//create index array for sorting. 
	thrust::device_vector<int> key_locations(num_value_rows);
	thrust::device_vector<uint32_t> d_unique_keys(num_value_rows);

	//Find count of unqiue keys - save location of where to find each key
	thrust::copy(d_hash_keys, d_hash_keys + num_key_rows,d_unique_keys.begin());
	thrust::copy(d_i.begin(), d_i.end(), key_locations.begin()); 
	thrust::pair<thrust::device_vector<uint32_t>::iterator, thrust::device_vector<int>::iterator> end = thrust::unique_by_key(d_unique_keys.begin(), d_unique_keys.end(), key_locations.begin());

	num_output_rows = end.second - key_locations.begin();

	//setup output arrays
	// output_keys = (int *)realloc(output_keys, num_output_rows*num_key_columns * sizeof(T));
	// output_values = (int *)realloc(output_values, num_output_rows*num_value_columns * sizeof(T));
	
	thrust::device_vector<T> d_column(num_value_rows);
	//copy back unique original keys to output array
	for (int i = 0; i<num_key_columns; i++){//i represents column of key 
		thrust::host_vector<T> h_column(key_columns + (i*num_key_rows), key_columns + ((i+1)*num_key_rows));
		d_column = h_column;
		thrust::permutation_iterator<ElementIterator,IndexIterator> data(d_column.begin(),key_locations.begin());
		thrust::copy_n(data, num_output_rows, output_keys+i*num_output_rows);
	}

	T* ones;
	cudaMalloc((void **) &ones, num_key_rows * sizeof(T));
	thrust::device_ptr<T> d_ones(ones);
	thrust::fill(d_ones, d_ones + num_key_rows, (T) 1);


	//the column is not sorted yet so use d_i to sort! 
	// note: is this vector initialized with di?
	thrust::device_vector<T> sorted_col(num_value_rows);

	//make device pointers for 
	T* out_ptr;
	cudaMalloc((void **) &out_ptr, num_output_rows * sizeof(T));
	thrust::device_ptr<T> d_output(out_ptr);	

	T* outkey_ptr;
	cudaMalloc((void **) &outkey_ptr, num_output_rows * sizeof(T));
	thrust::device_ptr<T> d_output_keys(outkey_ptr);

	//iterate though all columns of the matrix. Perfrom the operation corresponding to that column
	for (int i = 0; i<num_ops; i++){//i represents column of output
		//get this column of data. copy does [first, last) 
		int start = i*num_value_rows;
		
		//copy one column to device vecotr for calculation
		thrust::copy_n(value_columns + start,num_value_rows,d_column.begin());
		thrust::permutation_iterator<ElementIterator,IndexIterator> iter(d_column.begin(),d_i.begin());
		thrust::copy_n(iter, num_value_rows, sorted_col.begin());

		thrust::equal_to<T> eq;
		thrust::minimum<T> mn;
		thrust::maximum<T> mx;
		thrust::plus<T> pls;
		switch(ops[i]){
			case rmax:
				thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_rows, sorted_col.begin(), d_output_keys, d_output,eq, mx);
				break;
			case rmin:
				thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_rows, sorted_col.begin(), d_output_keys, d_output, eq, mn);
				break;
			case rsum:
				thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_rows, sorted_col.begin(), d_output_keys, d_output,eq,pls);
				break;
			case rcount:
				thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_rows, thrust::make_constant_iterator(1), d_output_keys, d_output, eq, pls);
				break;
			case rmean:
				T* sums_ptr;
				cudaMalloc((void **) &sums_ptr, num_output_rows * sizeof(T));
				thrust::device_ptr<T> d_output_sums(sums_ptr);
				//get count for each key
				thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_rows, thrust::make_constant_iterator(1), d_output_keys, d_output, eq, pls);
				//Get sum for each key
				thrust::reduce_by_key(d_hash_keys, d_hash_keys + num_key_rows, sorted_col.begin(), d_output_keys, d_output_sums,eq,pls);
				//Perform division: Sums/Counts
				thrust::divides<T> div;
				thrust::transform(d_output, d_output + num_output_rows, d_output_sums, d_output, div);
				break;
		}
		int output_start = i*num_output_rows;
		thrust::copy(d_output, d_output + num_output_rows, output_values + output_start);
		
	}
	thrust::device_free(d_output);
	thrust::device_free(d_output_keys);
	thrust::device_free(d_ones);
}
