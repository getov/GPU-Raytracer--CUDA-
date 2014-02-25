#include "WaterWaves.cuh"
#include "Scene.cuh"
#include "Util.cuh"

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
		double dx = 0;
		double dy = 0;

		for (int i = 0; i < 3; ++i)
		{                             // frequency                         intensity
			dx += sin(scene->waves * randomDouble(0.0, 20.0) * data.u) * randomDouble(0.0, 15.0) * strength; 
			dy += sin(scene->waves * randomDouble(0.0, 20.0) * data.v) * randomDouble(0.0, 15.0) * strength;
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