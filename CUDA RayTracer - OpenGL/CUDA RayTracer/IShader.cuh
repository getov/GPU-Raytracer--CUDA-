#ifndef I_SHADER_H
#define I_SHADER_H

#include "Vector3D.cuh"
#include "Color.cuh"
#include "IGeometry.cuh"
#include "cuda_renderer.cuh"

__device__ extern bool testVisibility(const Vector& from, const Vector& to);

class Shader
{
protected:
	Color _color;

public:
	__device__ 
	Shader(const Color& color);

	__device__ 
	virtual ~Shader() {}

	__device__ 
	virtual Color shade(Ray ray, const IntersectionData& data) = 0;
};

#endif