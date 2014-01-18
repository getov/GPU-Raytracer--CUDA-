#ifndef PHONG_H
#define PHONG_H

#include "IShader.cuh"

class Phong : public Shader
{
private:
	double exponent;
	float strength;

public:
	__device__
	Phong(const Color& diffuseColor, double exponent, float strength = 1);

	__device__
	Color shade(Ray ray, const IntersectionData& data);
};

#endif