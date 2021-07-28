
//Can be changed to any associative and commutative function with an identity
const float identity = 0.0f;
float f(float a, float b){
	return a+b;
}

float reduceGPU(float* input, unsigned int N, unsigned int type);