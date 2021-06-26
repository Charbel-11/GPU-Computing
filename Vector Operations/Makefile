NVCC    = nvcc
OBJ     = main.o kernel.o
EXE     = vectorOperations

default: $(EXE)

%.o: %.cu
	$(NVCC) -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

