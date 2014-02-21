/***************************************************************************
 *   Copyright (C) 2009-2013 by Veselin Georgiev, Slavomir Kaslev et al    *
 *   admin@raytracing-bg.net                                               *
 *																		   *
 *	 Contributor: Peter Getov											   *
 *																		   *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

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

	__device__ virtual ~Light() {}

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

	__device__ 
	virtual void setPosition(const Vector& pos) {}

	__device__
	virtual Vector getPosition() {}

	__device__
	virtual void regulatePower(const float& power) {}
};

class PointLight : public Light 
{
private:
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

	__device__ 
	void setPosition(const Vector& pos);

	__device__
	Vector getPosition();

	__device__
	void regulatePower(const float& power);
};

/// A rectangle light; uses a transform to position in space and change shape. The canonic
/// light is a 1x1 square, positioned in (0, 0, 0), pointing in the direction of -Y. The
/// light is one-sided (the +Y hemisphere doesn't get any light).
class RectLight : public Light
{
private:
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

class SpotLight : public Light
{
private:
	Vector pos;
	Vector dir;
	float innerAngle;
	float outerAngle;

public:
	__device__
	SpotLight(const Vector& position, const Vector& direction,
			  const Color& color, const float& power,
			  const float& innerAngle, const float& outerAngle);

	__device__ int getNumSamples();

	__device__
	void getNthSample(int sampleIdx, const Vector& shadePos, Vector& samplePos, Color& color);

	__device__
	bool intersect(const Ray& ray, double& intersectionDist);

	__device__
	float solidAngle(const Vector& x);
};

#endif