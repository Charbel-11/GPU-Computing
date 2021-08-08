#ifndef _SCAN_H_
#define _SCAN_H_

//Can be changed to any associative and commutative function with an identity
const int identity = 0;
template <typename T>
__host__ __device__ T f(T a, T b) { return a + b; }

template <typename T>
void scanGPU(const T* input, T* output, unsigned int N, unsigned int type, bool inclusive);

//Note that it is safe to have input = output as output is changed in the end only
template <typename T>
void scanGPUOnDevice(const T* input_d, T* output_d, unsigned int N, unsigned int type, bool inclusive);

#endif