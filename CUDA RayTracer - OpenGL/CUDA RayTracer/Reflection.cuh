#ifndef REFLECTION_H
#define REFLECTION_H

#include "IShader.cuh"

class Reflection : public Shader
{
private:
	double glossiness;
	int numSamples;

	__device__
	static void getRandomDiscPoint(double& x, double& y);

public:
	__device__
	Reflection(const Color& filter = Color(1, 1, 1), double glossiness = 1.0, int numSamples = 20);

	__device__
	Color shade(Ray ray, const IntersectionData& data);
};

#endif