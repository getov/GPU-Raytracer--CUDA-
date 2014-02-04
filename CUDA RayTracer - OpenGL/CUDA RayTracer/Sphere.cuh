#ifndef SPHERE_H
#define SPHERE_H

#include "IGeometry.cuh"

class Sphere : public Geometry
{
private:
	Vector m_center;
	float m_radius;

public:
	__device__ 
	Sphere(Vector& vec3, float radius) 
	{
		m_center = vec3;
		m_radius = radius;
	}

	__device__ 
	bool intersect(Ray ray, IntersectionData& data);
};

#endif