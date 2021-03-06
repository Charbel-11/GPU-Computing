#ifndef _REDUCTION_H_
#define _REDUCTION_H_

//Can be changed to any associative and commutative function with an identity
const int identity = 0;
template <typename T>
__host__ __device__ T f(T a, T b) { return a + b; }

template <typename T>
T reduceGPU(const T* input, unsigned int N, unsigned int type);

#endif