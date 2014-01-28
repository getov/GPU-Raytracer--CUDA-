#include "WaterWaves.cuh"

__device__
WaterWaves::WaterWaves()
	: strength(0)
{
}

__device__
WaterWaves::WaterWaves(const float& waveStrength)
	: strength(waveStrength)
{
}

__device__
void WaterWaves::modifyNormal(IntersectionData& data)
{
	if (strength > 0)
	{
		float freqX[3] = { 0.5, 1.21, 1.9 };
		float freqZ[3] = { 0.4, 1.13, 1.81 };
		float waveDensity = 0.2;
		float intensityX[3] = { 0.1, 0.08, 0.05 };
		float intensityZ[3] = { 0.1, 0.08, 0.05 };

		double dx = 0;
		double dy = 0;

		for (int i = 0; i < 3; ++i) 
		{
			dx += sin(waveDensity * freqX[i] * data.u) * intensityX[i] * strength; 
			dy += sin(waveDensity * freqZ[i] * data.v) * intensityZ[i] * strength;
		}

		data.normal += dx * data.dNdx + dy * data.dNdy;
		data.normal.normalize();
	}
}

__device__
Color WaterWaves::getTexColor(const Ray& ray, double u, double v, Vector& normal)
{
	return Color(0, 0, 0);
}