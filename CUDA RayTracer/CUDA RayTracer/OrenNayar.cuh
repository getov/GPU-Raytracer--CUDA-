#ifndef OREN_NAYAR_H
#define OREN_NAYAR_H

#include "IShader.cuh"

class OrenNayar : public Shader
{
private:
	float sigma;

public:
	__device__
	OrenNayar(const Color& diffuseColor, const float& roughness);

	__device__
	Color shade(Ray ray, const IntersectionData& data);
};

#endif