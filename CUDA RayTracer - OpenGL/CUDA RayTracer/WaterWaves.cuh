#ifndef WATER_WAVES_H
#define WATER_WAVES_H

#include "Texture.cuh"

class WaterWaves : public Texture
{
private:
	float strength;

public:
	__device__
	WaterWaves();

	__device__
	explicit WaterWaves(const float& waveStrength);

	__device__
	void modifyNormal(IntersectionData& data);

	__device__
	Color getTexColor(const Ray& ray, double u, double v, Vector& normal);
};

#endif