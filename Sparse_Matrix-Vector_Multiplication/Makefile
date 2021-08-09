NVCC    = nvcc
OBJ     = main.o SpMV_COO.o SpMV_CSR.o SpMV_ELL.o
EXE     = SpMV

default: $(EXE)

%.o: %.cu
	$(NVCC) -c -o $@ $<
	
$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
