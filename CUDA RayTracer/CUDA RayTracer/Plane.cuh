#ifndef PLANE_H
#define PLANE_H

#include "IGeometry.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Settings.cuh"

class Plane : public Geometry
{
private:
	double _y;
	double _limitX;
	double _limitZ;

public:
	__device__ 
	Plane(double y, double limitX = INF, double limitZ = INF);

	__device__ 
	bool intersect(Ray ray, IntersectionData& data);
};

#endif