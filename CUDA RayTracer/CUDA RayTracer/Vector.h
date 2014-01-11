#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <math.h>
#include <ostream>
#include <iomanip>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Vector
{
public:
	union
	{
		struct
		{
			double x, y, z;
		};
		double components[3];
	};

	__device__ __host__ Vector();
	__device__ __host__ Vector(double _x, double _y, double _z);

	__device__ __host__ void makeZero();
	__device__ __host__ double length() const;
	__device__ __host__ double lengthSqr() const;
	__device__ __host__ void scale(double multiplier);

	__device__ __host__ void operator *= (double multiplier);
	__device__ __host__ void operator += (const Vector& rhs);
	__device__ __host__ void operator /= (double divider);

	__device__ __host__ void normalize();
	__device__ __host__ void setLength(double newLength);
	__device__ __host__ double& operator[] (int index);
	__device__ __host__ const double& operator[] (int index) const;
};

__device__ __host__ inline Vector operator + (const Vector& a, const Vector& b)
{
	return Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline Vector operator - (const Vector& a, const Vector& b)
{
	return Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline Vector operator - (const Vector& a)
{
	return Vector(-a.x, -a.y, -a.z);
}

//* dot product
__device__ __host__ inline double operator * (const Vector& a, const Vector& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

//* dot product (functional form, to make it more explicit):
__device__ __host__ inline double dot(const Vector& a, const Vector& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

//* cross product
__device__ __host__ inline Vector operator ^ (const Vector& a, const Vector& b)
{
	return Vector(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

__device__ __host__ inline Vector operator * (const Vector& a, double multiplier)
{
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}

__device__ __host__ inline Vector operator * (double multiplier, const Vector& a)
{
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}

__device__ __host__ inline Vector operator / (const Vector& a, double divider)
{
	double multiplier = 1.0 / divider;
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}

__device__ __host__ inline Vector reflect(const Vector& ray, const Vector& norm)
{
	Vector result = ray - 2 * dot(ray, norm) * norm;
	result.normalize();
	return result;
}

__device__ __host__ inline Vector faceforward(const Vector& ray, const Vector& norm)
{
	if (dot(ray, norm) < 0)
	{
		return norm;
	}
	else
	{
		return -norm;
	}
}

__device__ __host__ inline Vector project(const Vector& v, int a, int b, int c)
{
	Vector result;
	result[a] = v[0];
	result[b] = v[1];
	result[c] = v[2];
	return result;
}

__device__ __host__ inline Vector unproject(const Vector& v, int a, int b, int c)
{
	Vector result;
	result[0] = v[a];
	result[1] = v[b];
	result[2] = v[c];
	return result;
}

struct Ray
{
	Vector start;
	Vector dir;

	bool debug;
	
	__device__ __host__ Ray();
	__device__ __host__ Ray(const Vector& _start, const Vector& _dir);
};

__device__ __host__ inline Ray project(const Ray& v, int a, int b, int c)
{
	return Ray(project(v.start, a, b, c), project(v.dir, a, b, c));
}

// iostream Vector print routine:
//__host__ __device__ inline std::ostream& operator << (std::ostream& os, const Vector& vec)
//{
//	os << "(" << std::fixed << std::setprecision(3) << vec.x << ", " << vec.y << ", " << vec.z << ")";
//	return os;
//}

#endif // VECTOR3D_H