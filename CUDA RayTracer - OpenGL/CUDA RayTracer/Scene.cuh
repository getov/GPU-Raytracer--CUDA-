#ifndef SCENE_H
#define SCENE_H

#include "custom_vector.cuh"
#include "IGeometry.cuh"
#include "IShader.cuh"
#include "Node.cuh"
#include "Light.cuh"

class Scene
{
public:
	__device__ Scene();
	__device__ ~Scene();

	pgg::vector<Geometry*> dev_geom;
	pgg::vector<Shader*> dev_shaders;
	pgg::vector<Texture*> dev_textures;
	pgg::vector<Node*> dev_nodes;
	pgg::vector<Light*> dev_lights;

	Color ambientLight;
};

#endif