#include "Scene.cuh"

__device__ Scene::Scene()
	: selectedNode(nullptr)
	, indexGeom(-1)
	, indexShader(-1)
	, isFogActive(false)
	, fogDensity(150.0)
	, fogColor(0.5, 0.5, 0.5)
{
	ambientLight = Color(0.2, 0.2, 0.2);
	waves = 0.2;
}

__device__ Scene::~Scene()
{
	for (int i = 0; i < dev_nodes.size(); ++i)
	{
		delete dev_geom[i];
		delete dev_shaders[i];
		delete dev_textures[i];
		delete dev_nodes[i];
	}

	for (int i = 0; i < dev_lights.size(); ++i)
	{
		delete dev_lights[i];
	}

	delete dev_cam;
}