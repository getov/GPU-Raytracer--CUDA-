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

	__device__  Vector(){}

	__device__  
	Vector(const double& _x, const double& _y, const double& _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	__device__
	void set(const double& _x, const double& _y, const double& _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	__device__  
	void makeZero()
	{
		x = y = z = 0.0;
	}

	__device__  
	double length() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	__device__  
	double lengthSqr() const
	{
		return (x * x + y * y + z * z);
	}

	__device__  
	void scale(const double& multiplier)
	{
		x *= multiplier;
		y *= multiplier;
		z *= multiplier;
	}

	__device__ 
	void operator *= (const double& multiplier)
	{
		scale(multiplier);
	}

	__device__  
	void operator += (const Vector& rhs)
	{
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
	}

	__device__  
	void operator /= (const double& divider)
	{
		scale(1.0 / divider);
	}

	__device__  
	void normalize()
	{
			double multiplier = 1.0 / length();
			scale(multiplier);
	}

	__device__  
	void setLength(const double& newLength)
	{
		scale(newLength / length());
	}

	__device__  
	double& operator[] (const int& index)
	{
		return components[index];
	}

	__device__  
	const double& operator[] (const int& index) const
	{
		return components[index];
	}
};

__device__  
inline Vector operator + (const Vector& a, const Vector& b)
{
	return Vector(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__  
inline Vector operator - (const Vector& a, const Vector& b)
{
	return Vector(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__  
inline Vector operator - (const Vector& a)
{
	return Vector(-a.x, -a.y, -a.z);
}

//* dot product
__device__ 
inline double operator * (const Vector& a, const Vector& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

//* dot product (functional form, to make it more explicit):
__device__  
inline double dot(const Vector& a, const Vector& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

//* cross product
__device__  
inline Vector operator ^ (const Vector& a, const Vector& b)
{
	return Vector(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

__device__  
inline Vector operator * (const Vector& a, double multiplier)
{
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}

__device__  
inline Vector operator * (const double& multiplier, const Vector& a)
{
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}

__device__  
inline Vector operator / (const Vector& a, const double& divider)
{
	double multiplier = 1.0 / divider;
	return Vector(a.x * multiplier, a.y * multiplier, a.z * multiplier);
}

__device__ 
inline Vector reflect(const Vector& ray, const Vector& norm)
{
	Vector result = ray - 2 * dot(ray, norm) * norm;
	result.normalize();
	return result;
}

__device__
inline Vector refract(const Vector& i, const Vector& n, float ior)
{
	float NdotI = float(dot(i, n));
	float k = 1 - (ior * ior) * (1 - NdotI * NdotI);
	if (k < 0)
	{
		return Vector(0, 0, 0);
	}
	return ior * i - (ior * NdotI + sqrt(k)) * n;
}

__device__  
inline Vector faceforward(const Vector& ray, const Vector& norm)
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

__device__  
inline Vector project(const Vector& v, const int& a, const int& b, const int& c)
{
	Vector result;
	result[a] = v[0];
	result[b] = v[1];
	result[c] = v[2];
	return result;
}

__device__  
inline Vector unproject(const Vector& v, const int& a, const int& b, const int& c)
{
	Vector result;
	result[0] = v[a];
	result[1] = v[b];
	result[2] = v[c];
	return result;
}

__device__
inline Vector normalize(const Vector& vec)
{
	double multiplier = 1.0 / vec.length();
	return vec * multiplier;
}

/// given an unit vector a, create an orhonormed system (a, b, c). Code is deterministic.
__device__
inline void orthonormedSystem(const Vector& a, Vector& b, Vector& c)
{
	Vector temp = Vector(1, 0, 0);

	if (fabs(dot(a, temp)) > 0.99)
	{
		temp = Vector(0, 1, 0);
		if (fabs(dot(a, temp)) > 0.99)
			temp = Vector(0, 0, 1);
	}

	b = a ^ temp;
	b.normalize();
	c = a ^ b;
	c.normalize();
}

struct Ray
{
	Vector start;
	Vector dir;
	int depth;

	bool debug;
	
	__device__  Ray()
	{
		debug = false;
	}

	__device__  
	Ray(const Vector& _start, const Vector& _dir)
	{
		start = _start;
		dir = _dir;
		debug = false;
		depth = 0;
	}
};

__device__  
inline Ray project(const Ray& v, const int& a, const int& b, const int& c)
{
	return Ray(project(v.start, a, b, c), project(v.dir, a, b, c));
}

#endif // VECTOR3D_H