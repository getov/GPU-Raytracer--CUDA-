#include "Scene.cuh"

__device__ Scene::Scene()
{
	ambientLight = Color(0.2, 0.2, 0.2);
}

__device__ Scene::~Scene()
{
	for (int i = 0; i < dev_nodes.size(); ++i)
	{
		delete dev_geom[i];
		delete dev_shaders[i];
		delete dev_textures[i];
		delete dev_nodes[i];

		printf("scene elements deleted\n");
	}

	for (int i = 0; i < dev_lights.size(); ++i)
	{
		delete dev_lights[i];

		printf("scene elements deleted\n");
	}

	delete dev_cam;
}