NVCC    = nvcc
OBJ     = main.o convolution.o
EXE     = convolution

default: $(EXE)

%.o: %.cu
	$(NVCC) -c -o $@ $<
	
$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)
