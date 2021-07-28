#ifndef _COMMON_H_
#define _COMMON_H_

//Can be changed to any associative and commutative function with an identity
const float identity = 0.0f;
__host__ __device__ float f(float a, float b);

float reduceGPU(float* input, unsigned int N, unsigned int type);

#endif