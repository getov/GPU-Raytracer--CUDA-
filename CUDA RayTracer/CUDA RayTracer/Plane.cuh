#ifndef PLANE_H
#define PLANE_H

#include "IGeometry.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Plane : public Geometry
{
private:
	double _y;

public:
	__device__ Plane(double y) 
		: _y(y) 
	{}

	__device__ bool intersect(Ray ray, IntersectionData& data);
};

#endif