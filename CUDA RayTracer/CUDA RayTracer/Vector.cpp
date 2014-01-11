#include "Vector.h"

__device__ Vector::Vector()
{
}

__device__ __host__ Vector::Vector(double _x, double _y, double _z)
{
	x = _x;
	y = _y;
	z = _z;
}

__device__ __host__ void Vector::makeZero()
{
	x = y = z = 0.0;
}

 __device__ __host__ double Vector::length() const
{
	return sqrt(x * x + y * y + z * z);
}

__device__ __host__ double Vector::lengthSqr() const
{
	return (x * x + y * y + z * z);
}

__device__ __host__ void Vector::scale(double multiplier)
{
	x *= multiplier;
	y *= multiplier;
	z *= multiplier;
}

__device__ __host__ void Vector::operator *= (double multiplier)
{
	scale(multiplier);
}

__device__ __host__ void Vector::operator += (const Vector& rhs)
{
	x += rhs.x;
	y += rhs.y;
	z += rhs.z;
}

__device__ __host__ void Vector::operator /= (double divider)
{
	scale(1.0 / divider);
}

__device__ __host__ void Vector::normalize()
{
		double multiplier = 1.0 / length();
		scale(multiplier);
}

__device__ __host__ void Vector::setLength(double newLength)
{
	scale(newLength / length());
}

__device__ __host__ double& Vector::operator[] (int index)
{
	return components[index];
}

__device__ __host__ const double& Vector::operator[] (int index) const
{
	return components[index];
}

__device__ __host__ Ray::Ray()
{
	debug = false;
}

__device__ __host__ Ray::Ray(const Vector& _start, const Vector& _dir)
{
	start = _start;
	dir = _dir;
	debug = false;
}