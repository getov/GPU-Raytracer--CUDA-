#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>
#include <math.h>
#include "Settings.cuh"
#include "curand.h"
#include "curand_kernel.h"
#include <thrust\random.h>
#include <ctime>
#include <string>
#include <vector>
using std::string;

__device__ 
inline double signOf(double x)
{ 
	return x > 0 ? +1 : -1; 
}

__device__ 
inline double sqr(double a) 
{ 
	return a * a;
}

__device__ 
inline double toRadians(double angle) 
{
	return angle / 180.0 * PI; 
} 

__device__ 
inline double toDegrees(double angle_rad)
{ 
	return angle_rad / PI * 180.0;
} 

__device__ __host__ 
inline int nearestInt(float x) 
{ 
	return (int) floor(x + 0.5f);
}

__device__ 
inline bool solveQuadraticEquation(float a, float b, float c, float& outX1, float& outX2)
{
	float discriminant = b*b - 4*a*c;
	if(discriminant < 0)
	{
		return false;
	}

	float sqrtD = sqrt(discriminant);

	outX1 = (-b + sqrtD)/(2*a);
	outX2 = (-b - sqrtD)/(2*a);

	return true;
}

__device__
inline float noise(unsigned x)
{
	x = (x<<13) ^ x;
	return ( 1.0 - ( (x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0);
}

__device__
inline float perlin(float x, float y)
{
	float res = 0.0f, amp = 1.0f;
	unsigned size = 1;
	for (int i = 0; i <6; i++, size*=2) {
		unsigned x0 = (unsigned) (x*size);
		unsigned y0 = (unsigned) (y*size);
		unsigned q = x0 + y0 *size;
		float fx = x*size - x0;
		float fy = y*size - y0;
		float nf = noise(q       )*(1.0f-fx)*(1.0f-fy) +
				noise(q+1     )*(     fx)*(1.0f-fy) +
				noise(q  +size)*(1.0f-fx)*(     fy) +
				noise(q+1+size)*(     fx)*(     fy);
		res += amp * nf;
		amp *= 0.72;
	}
	return res;
}

__device__
inline float randomFloat()
{
	thrust::default_random_engine randEngine(123u);
	thrust::uniform_real_distribution<float> generator;

	return (float)generator(randEngine);
}

__device__
inline double randomDouble(const double& lowerBound, const double& upperBound)
{
	thrust::default_random_engine randEngine(123u);
	thrust::uniform_real_distribution<double> generator(lowerBound, upperBound);

	return (double)generator(randEngine);
}

template <typename T>
__device__ T dev_min(const T& a, const T& b)
{
	return a < b ? a : b;
}

template <typename T>
__device__ T dev_max(const T& a, const T& b)
{
	return a > b ? a : b;
}

template <typename T>
__device__ void dev_swap(T& a, T& b)
{
	T temp(a);
	a = b;
	b = temp;
}

#endif // UTIL_H
