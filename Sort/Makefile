NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.o mergeSort.o radixSort.o
DEP_OBJ		= ../Merge/merge.o ../Scan/scan.o
EXE         = sort

default: $(EXE)

%.o: %.cu
	$(NVCC) -c -o $@ $<
	
$(EXE): $(OBJ) $(DEP_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) $(DEP_OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

