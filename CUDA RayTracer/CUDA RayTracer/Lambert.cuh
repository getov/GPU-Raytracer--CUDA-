#ifndef LAMBERT_H
#define LAMBERT_H

#include "IShader.cuh"

class Lambert : public Shader
{
public:
	__device__ Lambert(const Color& diffuseColor);
	__device__ Color shade(Ray ray, const IntersectionData& data);
};

#endif