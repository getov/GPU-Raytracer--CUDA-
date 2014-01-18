#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>
#include <math.h>
#include "Settings.cuh"
#include "curand.h"
#include "curand_kernel.h"

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

/// returns a random floating-point number in [0..1).
/// This is not a very good implementation. A better method is to be employed soon.
__device__
inline float randomFloat()
{
	//unsigned int N = 10000; // samples per thread

 //   unsigned int seed = thread_id;

 //   curandState s;
	//curandGenerateUniform curand_uniform 
	return rand() / (float) RAND_MAX;
}

#endif // UTIL_H
