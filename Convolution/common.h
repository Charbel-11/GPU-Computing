#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS)*2 + 1)

void convolutionGPU(float* input, float* output, float mask[MASK_DIM][MASK_DIM], unsigned int width, unsigned int height, unsigned int type);