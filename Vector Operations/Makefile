NVCC    = /usr/lib/cuda/bin/nvcc
OBJ     = main.o kernel.o
EXE     = vecmax


default: $(EXE)

%.o: %.cu
	$(NVCC) -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

