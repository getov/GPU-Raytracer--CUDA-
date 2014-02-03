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
inline float randomFloat()
{
	thrust::default_random_engine randEngine(123u);
	thrust::uniform_real_distribution<float> generator;

	return (float)generator(randEngine);
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

//
// parse a string, convert to double. If string is empty, return 0
static double getDouble(const string& s)
{
	double res;
	if (s == "") return 0;
	sscanf(s.c_str(), "%lf", &res);
	return res;
}

// parse a string, convert to int. If string is empty, return 0
static int getInt(const string& s)
{
	int res;
	if (s == "") return 0;
	sscanf(s.c_str(), "%d", &res);
	return res;
}

std::vector<string> tokenize(string s);
std::vector<string> split(string s, char separator);

//class c_string
//{
//private:
//	char* temp;
//
//public:
//	__device__
//	c_string();
//
//	__device__
//	~c_string();
//
//	__device__
//	int length(const char* s);
//
//	__device__
//	bool isspace(const char s);
//
//	__device__
//	char* substr(const char* s, int pos, int length);
//
//	__device__
//	int atoi(const char* s);
//};
//
//__device__
//vector<char*> tokenize(char* s);
//
//__device__
//vector<char*> split(const char* s, char separator);

#endif // UTIL_H
