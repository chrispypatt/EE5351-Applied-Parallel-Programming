NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = histogram
OBJ	        = histogram_cu.o harness.o ref.o util.o

default: $(EXE)

histogram_cu.o: opt_2dhisto.cu opt_2dhisto.h ref_2dhisto.h
	$(NVCC) -c -o $@ opt_2dhisto.cu $(NVCC_FLAGS)

harness.o: test_harness.cpp util.cpp ref_2dhisto.cpp ref_2dhisto.h
	$(NVCC) -c -o $@ test_harness.cpp $(NVCC_FLAGS) 

ref.o: test_harness.cpp util.cpp ref_2dhisto.cpp ref_2dhisto.h
	$(NVCC) -c -o $@ ref_2dhisto.cpp $(NVCC_FLAGS) 

util.o: test_harness.cpp util.cpp ref_2dhisto.cpp ref_2dhisto.h
	$(NVCC) -c -o $@ util.cpp $(NVCC_FLAGS) 


$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
