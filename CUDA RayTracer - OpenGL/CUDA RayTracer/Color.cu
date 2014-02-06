#include "Color.cuh"

__device__
unsigned char RGB_COMPRESS_CACHE[4097];

__device__
unsigned char SRGB_COMPRESS_CACHE[4097];

__device__
void precomputeColorCache()
{
	for (int i = 0; i <= 4096; i++)
	{
		RGB_COMPRESS_CACHE[i] = (unsigned char) convertTo8bit(i / 4096.0f);
	}

	for (int i = 0; i <= 4096; i++)
	{
		SRGB_COMPRESS_CACHE[i] = (unsigned char) convertTo8bit_sRGB(i / 4096.0f);
	}
}