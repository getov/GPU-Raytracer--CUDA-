#ifndef I_GEOMETRY_H
#define I_GEOMETRY_H

#include "Vector3D.cuh"

class Geometry;
struct IntersectionData 
{
	Vector p; //!< intersection point in the world-space
	Vector normal; //!< the normal of the geometry at the intersection point
	double dist; //!< before intersect(): the max dist to look for intersection; after intersect() - the distance found
	
	double u, v; //!< 2D UV coordinates for texturing, etc.
	
	Geometry* g; //!< The geometry which was hit
};

class Geometry
{
public:

	__device__ virtual ~Geometry() {}

	__device__ virtual bool intersect(Ray ray, IntersectionData& data) = 0;
};

#endif