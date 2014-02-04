#ifndef COLOR_H
#define COLOR_H

#include "device_launch_parameters.h"
#include "Util.cuh"

__device__ __host__ 
inline unsigned convertTo8bit(float x)
{
	if (x < 0) x = 0;
	if (x > 1) x = 1;
	return nearestInt(x * 255.0f);
}

__host__
inline unsigned convertTo8bit_sRGB(float x)
{
	const float a = 0.055f;
	if (x <= 0) return 0;
	if (x >= 1) return 255;
	// sRGB transform:
	if (x <= 0.0031308f)
		x = x * 12.02f;
	else
		x = (1.0f + a) * powf(x, 1.0f / 2.4f) - a;
	return nearestInt(x * 255.0f);
}

unsigned convertTo8bit_sRGB_cached(float x);

/// Represents a color, using floatingpoint components in [0..1]
struct Color
{
	// a union, that allows us to refer to the channels by name (::r, ::g, ::b),
	// or by index (::components[0] ...). See operator [].
	union {
		struct { float r, g, b; };
		float components[3];
	};
	//
	__device__ Color() {}

	__device__ 
	Color(float _r, float _g, float _b) //!< Construct a color from floatingpoint values
	{
		setColor(_r, _g, _b);
	}
	__device__ 
	explicit Color(unsigned rgbcolor) //!< Construct a color from R8G8B8 value like "0xffce08"
	{
		float divider = 1 / 255.0f;

		b = (rgbcolor & 0xff) * divider;
		g = ((rgbcolor >> 8) & 0xff) * divider;
		r = ((rgbcolor >> 16) & 0xff) * divider;
	}

	/// convert to RGB32, with channel shift specifications. The default values are for
	/// the blue channel occupying the least-significant byte
	__device__ __host__ 
	unsigned toRGB32(int redShift = 16, int greenShift = 8, int blueShift = 0)
	{
		unsigned ir = convertTo8bit(r);
		unsigned ig = convertTo8bit(g);
		unsigned ib = convertTo8bit(b);
		return (ib << blueShift) | (ig << greenShift) | (ir << redShift);
	}

	/// make black
	__device__ 
	void makeZero(void)
	{
		r = g = b = 0;
	}

	/// set the color explicitly
	__device__ 
	void setColor(float _r, float _g, float _b)
	{
		r = _r;
		g = _g;
		b = _b;
	}

	/// get the intensity of the color (direct)
	__device__ 
	float intensity(void)
	{
		return (r + g + b) * 0.3333333;
	}

	/// get the perceptual intensity of the color
	__device__
	float intensityPerceptual(void)
	{
		return (r * 0.299 + g * 0.587 + b * 0.114);
	}

	/// Accumulates some color to the current
	__device__ 
	void operator += (const Color& rhs)
	{
		r += rhs.r;
		g += rhs.g;
		b += rhs.b;
	}

	/// multiplies the color
	__device__ 
	void operator *= (float multiplier)
	{
		r *= multiplier;
		g *= multiplier;
		b *= multiplier;
	}

	/// divides the color
	__device__ 
	void operator /= (float divider)
	{
		r /= divider;
		g /= divider;
		b /= divider;
	}
	
	__device__ 
	inline const float& operator[] (int index) const
	{
		return components[index];
	}
	
	__device__ 
	inline float& operator[] (int index)
	{
		return components[index];
	}
};

/// adds two colors
__device__ 
inline Color operator + (const Color& a, const Color& b)
{
	return Color(a.r + b.r, a.g + b.g, a.b + b.b);
}

/// subtracts two colors
__device__
inline Color operator - (const Color& a, const Color& b)
{
	return Color(a.r - b.r, a.g - b.g, a.b - b.b);
}

/// multiplies two colors
__device__ 
inline Color operator * (const Color& a, const Color& b)
{
	return Color(a.r * b.r, a.g * b.g, a.b * b.b);
}

/// multiplies a color by some multiplier
__device__ 
inline Color operator * (const Color& a, float multiplier)
{
	return Color(a.r * multiplier, a.g * multiplier, a.b * multiplier);
}

/// multiplies a color by some multiplier
__device__ 
inline Color operator * (float multiplier, const Color& a)
{
	return Color(a.r * multiplier, a.g * multiplier, a.b * multiplier);
}

/// divides some color
__device__ 
inline Color operator / (const Color& a, float divider)
{
	return Color(a.r / divider, a.g / divider, a.b / divider);
}

#endif // COLOR_H
