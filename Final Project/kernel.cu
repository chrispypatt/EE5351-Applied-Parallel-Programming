#define EMPTY 0xffffffff

__device__ bool is_equal(int *key_columns, int a, int b, int num_key_columns, int num_key_rows)
{
  for (int i = 0; i < num_key_columns, ++i) 
    if (key_columns[i*num_key_rows + a] != key_columns[i*num_key_rows + b]) return false;
  return true;
}


__device__ void perform_op(int bucket, int *hash_table_aggregate, int hash_table_length, int i, reductionType op, int val)
{
  switch (op) {
  case rmax:
    atomicMax(&hash_table_aggregate[i * hash_table_length + bucket], val);
  case rmin:
    atomicMin(&hash_table_aggregate[i * hash_table_length + bucket], val);
  case rcount:
    atomicAdd(&hash_table_aggregate[i * hash_table_length + bucket], 1);
  case rsum:
    atomicAdd(&hash_table_aggregate[i * hash_table_length + bucket], val);
  }
}
__global__ void insert_and_aggregate(int *key_columns, int num_key_columns, int num_key_rows,
				     int *val_columns, int num_val_columns, int num_val_rows,
				     reductionType *op_columns, int num_op_columns,
				     uint32_t *hash_table_keys, int *hash_table_aggregate, 
				     int hash_table_length, int *insert_result)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t hashed = hash(key_columns, num_key_columns, num_key_rows);
  int bucket = hashed % hash_table_length;
  int attempts = 0;
  int result = 0;
  while (attempts < hash_table_length) {
    uint32_t val = hash_table_keys[bucket];
    if (val == EMPTY) {
      // try insert key
      uint32_t old = atomicCAS(&hash_table_keys[bucket], idx, EMPTY);
      if (old != EMPTY) {
	if (!is_equal(key_columns, idx, old, num_key_columns, num_key_rows)) result = 1; // lost race condition and collision
      }
    
    }
    else if (!is_equal(key_columns, idx, val, num_key_columns, num_key_rows)) result = 1; // collision
    if (result == 0) { // no collision, update aggregate
      for (int i < 0; i < num_op_columns; ++i) {
	perform_op(bucket, hash_table_aggregate, hash_table_length, i, op_columns[i], val_columns[i * num_val_rows + idx]);
      }
      break; // while loop
    }
    ++attempts;
    bucket = (bucket+1) % hash_table_length;
  }
  if ((result == 1) && (attempts == hash_table_length)) { // failed to insert
    atomicAdd(insert_result, 1);
  }
}
