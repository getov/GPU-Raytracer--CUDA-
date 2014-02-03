#ifndef LIGHT_H
#define LIGHT_H

#include "Color.cuh"
#include "Vector3D.cuh"
#include "Transform.cuh"

class Light
{
protected:
	Color m_color;
	float m_power;

public:
	__device__ Light()
	{
		m_color.makeZero();
		m_power = 0.0;
	}

	__device__ Light(const Color& col, const float& power)
		: m_color(col)
		, m_power(power)
	{
	}

	__device__ 
		virtual ~Light() {}

	__device__ Color getColor() const
	{
		return m_color * m_power;
	}

	/// get the number of samples this light requires (must be strictly positive)
	__device__ virtual int getNumSamples() = 0;

	/**
	 * gets the n-th sample
	 * @param sampleIdx - a sample index: 0 <= sampleIdx < getNumSamples().
	 * @param shadePos  - the point we're shading. Can be used to modulate light power if the
	 *                    light doesn't shine equally in all directions.
	 * @param samplePos [out] - the generated light sample position
	 * @param color [out] - the generated light "color". This is usually has large components (i.e.,
	 *                      it's base color * power
	 */
	__device__
	virtual void getNthSample(int sampleIdx, const Vector& shadePos, Vector& samplePos, Color& color) = 0;

	/**
	 * intersects a ray with the light. The param intersectionDist is in/out;
	 * it's behaviour is similar to Intersectable::intersect()'s treatment of distances.
	 * @retval true, if the ray intersects the light, and the intersection distance is smaller
	 *               than the current value of intersectionDist (which is updated upon return)
	 * @retval false, otherwise.
	 */
	__device__
	virtual bool intersect(const Ray& ray, double& intersectionDist) = 0;

	__device__
	virtual float solidAngle(const Vector& x) = 0;

	__device__
	virtual void beginFrame() {}
};

class PointLight: public Light 
{
	Vector pos;
public:
	__device__
	PointLight(const Vector& position, const Color& color, const float& power);

	__device__ int getNumSamples();

	__device__
	void getNthSample(int sampleIdx, const Vector& shadePos, Vector& samplePos, Color& color);

	__device__
	bool intersect(const Ray& ray, double& intersectionDist);

	__device__
	float solidAngle(const Vector& x);
};

/// A rectangle light; uses a transform to position in space and change shape. The canonic
/// light is a 1x1 square, positioned in (0, 0, 0), pointing in the direction of -Y. The
/// light is one-sided (the +Y hemisphere doesn't get any light).
class RectLight: public Light
{
	Transform transform;
	int xSubd, ySubd;
	float area;
	Vector center;

public:
	__device__ RectLight();

	__device__
	RectLight(const Vector& translate, const Vector& rotate, const Vector& scale,
			  const Color& color, const float& power, int xSubd = 2, int ySubd = 2);

	__device__ void beginFrame();

	__device__ int getNumSamples();

	__device__
	void getNthSample(int sampleIdx, const Vector& shadePos, Vector& samplePos, Color& color);

	__device__
	bool intersect(const Ray& ray, double& intersectionDist);

	__device__
	float solidAngle(const Vector& x);
};

#endif