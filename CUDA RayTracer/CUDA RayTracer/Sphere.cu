#include "Sphere.cuh"
#include "Util.cuh"

__device__ bool Sphere::intersect(Ray ray, IntersectionData& data)
{
	Vector sphereSpaceOrigin = ray.start - m_center;

	float a = ray.dir.lengthSqr();
	float b = 2 * dot(sphereSpaceOrigin, ray.dir);
	float c = sphereSpaceOrigin.lengthSqr() - m_radius * m_radius;

	float x1, x2;
	if(solveQuadraticEquation(a, b, c, x1, x2))
	{
		if(x1 < 0 && x2 < 0)
		{
			//behind the ray
			return false;
		}

		float solution;
		if(x1 < 0 && x2 >= 0)
		{
			// start within the sphere
			// single intersection point
			solution = x2;
		}
		else if(x1 >= 0 && x2 < 0)
		{
			// start within the sphere
			// single intersection point
			solution = x1;
		}
		else
		{
			// two intersection points
			// choose the closest
			solution = x1 < x2 ? x1 : x2;
		}

		if (solution > data.dist)
		{
			return false;
		}

		data.p = ray.start + solution * ray.dir;
		data.dist = solution;
		data.normal = data.p - m_center;
		data.normal.normalize();

		Vector relative = data.p - sphereSpaceOrigin;
		data.u = toDegrees(std::acos(relative.y));
		data.v = toDegrees(std::atan2(relative.z, relative.x));

		return true;
	}
	
	return false;
}