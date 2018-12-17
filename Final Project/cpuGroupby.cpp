//
//  cpuGroupby.cpp
//  RAPIDS
//
//  Created by Aaron on 11/27/18.
//  Copyright © 2018 Aaron Nightingale. All rights reserved.
//

#include <iostream>
#include "cpuGroupby.h"
#include <time.h>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>

void cpuGroupby::fillRand(int distinctKeys, int distinctVals) {
    srand((unsigned int)time(NULL));
    for (int cRow=0; cRow<num_key_rows; cRow++) {
        //Fill the key columns in a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            key_columns[keyIdx*num_key_rows + cRow] = rand() % (distinctKeys+1);
        }
        //Fill the value columns in a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            value_columns[valIdx*num_value_rows + cRow] = rand() % (distinctVals+1);
        }
    }
}

void cpuGroupby::allocResultArray() {
    output_keys = (int *)malloc(sizeof(int)*numGroups*num_key_columns);
    output_values = (int *)malloc(sizeof(int)*numGroups*num_value_columns);
}

void cpuGroupby::freeResults() {
    free(output_keys);
    free(output_values);
}

// Sorting Functions
void cpuGroupby::sort() {
    for (int i = 0; i < num_key_rows-1; i++) {
        // Last i elements are already in place
        for (int j = 0; j < num_key_rows-i-1; j++) {
            if (!nextKeyBigger(j)) {
                // Swap j and j+1
                swapAtRow(j);
            }
        }
    }
}

// QUICKSORT STUFF
// Based on https://www.geeksforgeeks.org/quick-sort/
void cpuGroupby::quickSort(int* array, int lowIdx, int highIdx) {
    if (lowIdx < highIdx) {
        /* pi is partitioning index, arr[pi] is now
         at right place */
        int pi = partition(array, lowIdx, highIdx);
        
        quickSort(array, lowIdx, pi - 1);  // Before pi
        quickSort(array, pi + 1, highIdx); // After pi
    }
}

void cpuGroupby::libsort() {
  std::vector<int> idx(num_key_rows);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), 
	    [=] (const int idx1, const int idx2) {
	      for (int i = 0; i < num_key_columns; ++i) {
		int data1 = key_columns[i * num_key_rows + idx1];
		int data2 = key_columns[i * num_key_rows + idx2];
		if (data1 > data2) return false;
		if (data1 < data2) return true;
	      }
	      return false;
	    });
  std::vector<int> new_row(num_key_rows);
  for (int i = 0; i < num_key_columns; ++i) {
    for (int j = 0; j < num_key_rows; ++j) {
      new_row[j] = key_columns[i * num_key_rows + idx[j]];
    }
    std::copy(new_row.begin(), new_row.end(), key_columns + i * num_key_rows);
  }
  for (int i = 0; i < num_value_columns; ++i) {
    for (int j = 0; j < num_value_rows; ++j) {
      new_row[j] = value_columns[i * num_value_rows + idx[j]];
    }
    std::copy(new_row.begin(), new_row.end(), value_columns + i * num_value_rows);
  }
}

int cpuGroupby::partition (int* array, int lowIdx, int highIdx) {
    // pivot (Element to be placed at right position)
    int pivotIdx = highIdx;
    
    int i = (lowIdx - 1);  // Index of smaller element
    for (int j = lowIdx; j <= highIdx-1; j++) {
        // If current element is smaller than or
        // equal to pivot
        if ( keyAtFirstIndexIsBigger(pivotIdx,j) ) {
            i++;    // increment index of smaller element
            swapValuesAtRows(i, j);
        }
    }
    
    swapValuesAtRows(i+1, highIdx);
    return (i + 1);
}

void cpuGroupby::swapValuesAtRows(int rowOne, int rowTwo) {
    int tempVal;
    //Swap the keys
    for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
        tempVal = key_columns[keyIdx*num_key_rows + rowOne];
        key_columns[keyIdx*num_key_rows + rowOne] = key_columns[keyIdx*num_key_rows + rowTwo];
        key_columns[keyIdx*num_key_rows + rowTwo] = tempVal;
    }
    //Swap the values
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
        tempVal = value_columns[valIdx*num_value_rows + rowOne];
        value_columns[valIdx*num_value_rows +rowOne]=value_columns[valIdx*num_value_rows +rowTwo];
        value_columns[valIdx*num_value_rows + rowTwo] = tempVal;
    }
}

bool cpuGroupby::keyAtFirstIndexIsBigger(int rowOne, int rowTwo) {
    int keyIdx=0;
    for (keyIdx=0; keyIdx<num_key_columns && key_columns[keyIdx*num_key_rows + rowOne] == key_columns[keyIdx*num_key_rows + rowTwo]; keyIdx++);
    // fix for equal keys
    if(keyIdx >= num_key_columns) {
        return false;
    }
    
    //see if key at cRow is greater than key at cRow+1
    if (key_columns[keyIdx*num_key_rows + rowOne] > key_columns[keyIdx*num_key_rows + rowTwo]) {
        return true;
    } else {
        return false;
    }
}
// END QUICKSORT STUFF

bool cpuGroupby::nextKeyBigger(int cRow) {
    int keyIdx=0;
    
    for (keyIdx=0; keyIdx<num_key_columns && key_columns[keyIdx*num_key_rows + cRow] == key_columns[keyIdx*num_key_rows + cRow+1]; keyIdx++);
    // fix for equal keys
    if(keyIdx >= num_key_columns) {
    	return false;
    }
    
    //see if key at cRow is greater than key at cRow+1
    if (key_columns[keyIdx*num_key_rows + cRow] > key_columns[keyIdx*num_key_rows + cRow+1]) {
        return false;
    } else {
        return true;
    }
}

void cpuGroupby::swapAtRow(int cRow) {
    int tempVal;
    //Swap the keys
    for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
        tempVal = key_columns[keyIdx*num_key_rows + cRow];
        key_columns[keyIdx*num_key_rows + cRow] = key_columns[keyIdx*num_key_rows + cRow+1];
        key_columns[keyIdx*num_key_rows + cRow+1] = tempVal;
    }
    //Swap the values
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
        tempVal = value_columns[valIdx*num_value_rows + cRow];
        value_columns[valIdx*num_value_rows + cRow]=value_columns[valIdx*num_value_rows + cRow+1];
        value_columns[valIdx*num_value_rows + cRow+1] = tempVal;
    }
}

void cpuGroupby::getNumGroups() {
    // Start with one group, each boundry is a "split" adding another group.
    numGroups = 1;
    // Store the start row of each group here
    tempCol[0] = 0;
    for (int cRow=1; cRow<num_key_rows; cRow++) {
        //Loop through all key columns
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (key_columns[keyIdx*num_key_rows+cRow] != key_columns[keyIdx*num_key_rows+cRow-1]) {
                // New Group found
                tempCol[numGroups] = cRow;
                numGroups++;
                keyIdx = num_key_columns+1; //break the for loop
            }
        }
    }
    cout << "numGroups: " << numGroups << endl;
}

void cpuGroupby::getGroupPtr() {
    // Allocate the group pointer array
    groupPtr = (int*) malloc(sizeof(int)*numGroups);
    
    // Fill it
    for (int i=0; i<numGroups; i++) {
        groupPtr[i] = tempCol[i];
    }
}

// Groupby functions
void cpuGroupby::groupby() {
    //max, min, sum, count, and arithmetic mean
    //Init reduction operation list (rmax for testing)
    for (int i=0; i<num_value_columns; i++) {
        ops[i] = rcount;
    }

    //sort();
    //quickSort(key_columns, 0, num_key_rows);
    libsort();
    
    getNumGroups();
    // printData();
    
    allocResultArray();
    getGroupPtr();
    writeOutputKeys();
    doReductionOps();
    printResults();
}

void cpuGroupby::doReductionOps() {
    for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
        switch (ops[valIdx]) {
            case rmax:
                rMax(valIdx);
                break;
            case rmin:
                rMin(valIdx);
                break;
            case rmean:
                rMean(valIdx);
                break;
            case rcount:
                rCount(valIdx);
                break;
            case rsum:
                rSum(valIdx);
                break;
            default:
                rMax(valIdx);
                break;
        }
    }
}

void cpuGroupby::rMax(int valIdx) {
    int maximum = -999999999;
    int tempVal;
    
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        maximum = -999999999;
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            tempVal = value_columns[ valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
            if (tempVal>maximum) {
                maximum = tempVal;
            }
        }
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = maximum;
    }
    
    //Handeling the final group
    maximum = -999999999;
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        tempVal = value_columns[ valIdx*num_value_rows + subIdx ];
        if (tempVal>maximum) {
            maximum = tempVal;
        }
    }
    // Copy values to the output array
    output_values[valIdx*numGroups + numGroups-1] = maximum;
}

void cpuGroupby::rMin(int valIdx) {
    int minimum = 999999999;
    int tempVal;
    
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        minimum = 999999999;
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            tempVal = value_columns[ valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
            if (tempVal<minimum) {
                minimum = tempVal;
            }
        }
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = minimum;
    }
    
    //Handeling the final group
    minimum = 999999999;
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        tempVal = value_columns[ valIdx*num_value_rows + subIdx ];
        if (tempVal<minimum) {
            minimum = tempVal;
        }
    }
    // Copy values to the output array
    output_values[valIdx*numGroups + numGroups-1] = minimum;
}

void cpuGroupby::rMean(int valIdx) {
    int sum=0;
    int mean=0;
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        sum = 0;
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            sum += value_columns[valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
        }
        // Copy values to the output array
        mean = sum / (groupPtr[groupIdx]-groupPtr[groupIdx-1]);
        output_values[valIdx*numGroups + groupIdx-1] = mean;
    }
    
    //Handeling the final group
    sum = 0;
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        sum += value_columns[ valIdx*num_value_rows + subIdx ];
    }
    // Copy values to the output array
    mean = sum / (num_value_rows-groupPtr[numGroups-1]);
    output_values[valIdx*numGroups + numGroups-1] = mean;
}

void cpuGroupby::rSum(int valIdx) {
    int sum=0;
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        sum = 0;
        for (int subIdx=0; subIdx<groupPtr[groupIdx]-groupPtr[groupIdx-1]; subIdx++) {
            sum += value_columns[valIdx*num_value_rows + groupPtr[groupIdx-1]+subIdx ];
        }
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = sum;
    }
    
    //Handeling the final group
    sum = 0;
    for (int subIdx=groupPtr[numGroups-1]; subIdx<num_value_rows; subIdx++) {
        sum += value_columns[ valIdx*num_value_rows + subIdx ];
    }
    // Copy values to the output array
    output_values[valIdx*numGroups + numGroups-1] = sum;
}

void cpuGroupby::rCount(int valIdx) {
    int count = 0;
    for (int groupIdx=1; groupIdx<numGroups; groupIdx++) {
        count = groupPtr[groupIdx]-groupPtr[groupIdx-1];
        
        // Copy values to the output array
        output_values[valIdx*numGroups + groupIdx-1] = count;
    }
    
    // Handling the final group
    count = num_value_rows-groupPtr[numGroups-1];
    output_values[valIdx*numGroups + numGroups-1] = count;
}

void cpuGroupby::writeOutputKeys() {
    // Copy each unique key to the output.
    int rowIdx;
    for (int groupIdx=0; groupIdx<numGroups; groupIdx++) {
        rowIdx = groupPtr[groupIdx];
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            output_keys[keyIdx*numGroups + groupIdx] = key_columns[keyIdx*num_key_rows + rowIdx];
        }
    }
}


// Debug / Printing Functions
void cpuGroupby::printData() {
    cout << "Printing Data..." << endl;
    
    for (int cRow=0; cRow<num_key_rows; cRow++) {
        //print keys for a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (keyIdx == 0) {
                cout << "{";
            }
            cout << key_columns[num_key_rows*keyIdx + cRow];
            if(keyIdx != num_key_columns-1) {
                cout << ":";
            } else {
                cout << "}:";
            }
        }
        // Print values for a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            if (valIdx == 0) {
                cout << "{";
            }
            cout << value_columns[num_value_rows*valIdx + cRow];
            if(valIdx != num_value_columns-1) {
                cout << ":";
            } else {
                cout << "}";
            }
        }
        cout << endl;
    }
    
    cout << "End Printing Data" << endl << endl;
}

void cpuGroupby::printResults() {
    cout << "Printing Results..." << endl;
    
    for (int cRow=0; cRow<numGroups; cRow++) {
        //print keys for a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (keyIdx == 0) {
                cout << "{";
            }
            cout << output_keys[numGroups*keyIdx + cRow];
            if(keyIdx != num_key_columns-1) {
                cout << ":";
            } else {
                cout << "}:";
            }
        }
        // Print values for a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            if (valIdx == 0) {
                cout << "{";
            }
            cout << output_values[numGroups*valIdx + cRow];
            if(valIdx != num_value_columns-1) {
                cout << ":";
            } else {
                cout << "}";
            }
        }
        cout << endl;
    }
    
    cout << "End Printing Results" << endl;
}

void cpuGroupby::printGPUResults(int* GPU_output_keys, int* GPU_output_values){
    cout << "Printing GPU Results..." << endl;
    
    for (int cRow=0; cRow<numGroups; cRow++) {
        //print keys for a row
        for (int keyIdx=0; keyIdx<num_key_columns; keyIdx++) {
            if (keyIdx == 0) {
                cout << "{";
            }
            cout << GPU_output_keys[numGroups*keyIdx + cRow];
            if(keyIdx != num_key_columns-1) {
                cout << ":";
            } else {
                cout << "}:";
            }
        }
        // Print values for a row
        for (int valIdx=0; valIdx<num_value_columns; valIdx++) {
            if (valIdx == 0) {
                cout << "{";
            }
            cout << GPU_output_values[numGroups*valIdx + cRow];
            if(valIdx != num_value_columns-1) {
                cout << ":";
            } else {
                cout << "}";
            }
        }
        cout << endl;
    }
    
    cout << "End GPU Printing Results" << endl;
}

// To do: Verify function w GPU code
bool cpuGroupby::validGPUResult(int* GPUKeys, int* GPUValues, int GPUOutputRows) {
    //ASSUMING THE GPU RESULT IS SORTED
    if (GPUOutputRows != numGroups) {
        cout << "FAILED - CPU Rows: " << numGroups << " GPU Rows: " << GPUOutputRows << endl;
        return false;
    }
    // cout << "GPU:CPU"<<endl;
    for (int i=0; i<num_value_columns*numGroups; i++) {
        // cout << GPUValues[i] << ":" << output_values[i] << endl;
        if (GPUValues[i] != output_values[i]) {
            cout << "FAILED - CPU data != GPU data " << endl;
            return false;
        }
    }
    cout << "PASSED - CPU data == GPU data " << endl;   
    return true;
}

// Contructor & Destructor Funcitons
cpuGroupby::~cpuGroupby() {
    //Free the allocated memory
    free(key_columns);
    free(value_columns);
    free(tempCol);
    free(ops);
    free(output_keys);
    free(output_values);
}

cpuGroupby::cpuGroupby(int numKeys, int numValues, int numRows) {
    // Save the arguments
    num_key_columns = numKeys;
    num_key_rows = numRows;
    num_value_columns = numValues;
    num_value_rows = numRows;
    num_ops = numValues;
    
    // Allocate key & value arrays
    key_columns = (int *)malloc(sizeof(int)*num_key_columns*num_key_rows);
    value_columns = (int *)malloc(sizeof(int)*num_value_columns*num_value_rows);
    tempCol = (int *)malloc(sizeof(int)*num_value_rows);
    ops = (reductionType*)malloc(sizeof(reductionType)*num_ops);
}



