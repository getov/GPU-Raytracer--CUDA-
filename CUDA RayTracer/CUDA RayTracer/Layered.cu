#include "Layered.cuh"

__device__
void Layered::addLayer(Shader* shader, const Color& blend, Texture* texture)
{
	Layer& l = layers[numLayers];
	l.shader = shader;
	l.blend = blend;
	l.texture = texture;
	numLayers++;
}

__device__
Color Layered::shade(Ray ray, const IntersectionData& data)
{
	Color result(0, 0, 0);
	Vector N = data.normal;
	for (int i = 0; i < numLayers; i++) {
		Layer& l = layers[i];
		Color opacity = l.texture ?
			l.texture->getTexColor(ray, data.u, data.v, N) : l.blend;
		Color transparency = Color(1, 1, 1) - opacity;
		result = transparency * result + opacity * l.shader->shade(ray, data);
	}
	return result;
}