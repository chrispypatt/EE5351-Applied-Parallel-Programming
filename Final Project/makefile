# Makefile for GPU GroupBy Project
# EE-5351 Fall 2018
NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" --relocatable-device-code true
CXX_FLAGS   = -std=c++11
ifdef dbg
	NVCC_FLAGS  += -g -G 
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = groupby
OBJ	        = main.o cpuGroupby.o groupby.o HashFunc.o 

default: $(EXE)

main.o: main.cu cpuGroupby.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

HashFunc.o: HashFunc.cu HashFunc.cuh
	$(NVCC) -c -o $@ HashFunc.cu $(NVCC_FLAGS)

groupby.o: groupby.cu 
	$(NVCC) -c -o $@ groupby.cu $(NVCC_FLAGS)

cpuGroupby.o: cpuGroupby.cpp cpuGroupby.h
	$(NVCC) -c -o $@ cpuGroupby.cpp $(NVCC_FLAGS) $(CXX_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
