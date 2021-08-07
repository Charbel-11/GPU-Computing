#ifndef _SCAN_H_
#define _SCAN_H_

//Can be changed to any associative and commutative function with an identity
const double identity = 0.0;
__host__ __device__ double f(double a, double b);

void scanGPU(double* input, double* output, unsigned int N, unsigned int type, bool inclusive);

#endif