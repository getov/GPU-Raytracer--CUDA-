#ifndef LAYERED_H
#define LAYERED_H

#include "IShader.cuh"
#include "Texture.cuh"

class Layered: public Shader
{
	struct Layer
	{
		Shader* shader;
		Color blend;
		Texture* texture;
	};

	static const int MAX_LAYERS = 32;
	Layer layers[MAX_LAYERS];
	int numLayers;
public:
	__device__
	Layered();

	__device__
	void addLayer(Shader* shader, const Color& blend, Texture* texture = NULL);

	__device__
	Color shade(Ray ray, const IntersectionData& data);
};

#endif