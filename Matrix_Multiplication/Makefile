NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.o matrixMult.o
EXE         = matrixMultiplication


default: $(EXE)

%.o: %.cu
	$(NVCC) -c -o $@ $<
	
$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

