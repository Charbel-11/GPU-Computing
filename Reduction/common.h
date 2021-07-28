#ifndef _COMMON_H_
#define _COMMON_H_

//Can be changed to any associative and commutative function with an identity
const double identity = 0.0;
__host__ __device__ double f(double a, double b);

double reduceGPU(double* input, unsigned int N, unsigned int type);

#endif