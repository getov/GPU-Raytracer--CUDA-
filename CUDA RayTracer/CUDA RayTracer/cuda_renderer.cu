#ifndef __CUDACC__  
    #define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_renderer.cuh"
#include "Vector3D.cuh"
#include "Matrix.cuh"
#include "Color.cuh"
#include "Camera.cuh"
#include "IGeometry.cuh"
#include "Plane.cuh"
#include "Node.cuh"
#include "IShader.cuh"
#include "Lambert.cuh"
#include "Sphere.cuh"

__device__ 
bool testVisibility(Node** dev_nodes, const IntersectionData& data)
{
	//Vector to = lightPos;
	// Vector from = ray.start
	Ray ray;
	ray.start = data.p + data.normal * 1e-3;

	ray.dir = lightPos - ray.start;
	ray.dir.normalize();
	
	IntersectionData temp;
	temp.dist = (lightPos - ray.start).length();
	
	for (int i = 0; i < GEOM_COUNT; ++i)
	{
		if (dev_nodes[i]->geom->intersect(ray, temp))
		{
			return false;
		}
	}

	return true;
}

__global__ 
void initializeScene(Camera* dev_cam, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes)
{	
	dev_cam->yaw = 0;
	dev_cam->pitch = -30;
	dev_cam->roll = 0;
	dev_cam->fov = 90;
	dev_cam->aspect = 4.0 / 3.0;
	dev_cam->pos = Vector(0, 160, -50);

	dev_cam->beginFrame();
	
	lightPos = Vector(-90, 200, 150);
	lightColor = Color(1, 1, 1);
	lightPower = 50000;
	ambientLight = Color(0.2, 0.2, 0.2);

	dev_geom[0] = new Plane(5);
	dev_shaders[0] = new Lambert(Color(0, 1, 0));
	dev_nodes[0] = new Node(dev_geom[0], dev_shaders[0]);

	dev_geom[1] = new Sphere(Vector(-80, 40, 180), 50.0);
	dev_shaders[1] = new Lambert(Color(1.0, 1.0, 0.0));
	dev_nodes[1] = new Node(dev_geom[1], dev_shaders[1]);

	dev_geom[2] = new Sphere(Vector(0, 30, 200), 50.0);
	dev_shaders[2] = new Lambert(Color(1.0, 0.0, 0.0));
	dev_nodes[2] = new Node(dev_geom[2], dev_shaders[2]);
}

__device__ 
Color raytrace(Ray ray, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes)
{
	IntersectionData data;
	Node* closestNode = nullptr;

	data.dist = 1e99;

	for (int i = 0; i < GEOM_COUNT; i++)
	{
		if (dev_nodes[i]->geom->intersect(ray, data))
		{
			closestNode = dev_nodes[i];
		}
	}

	if (!closestNode)
	{
		return Color(0, 0, 0);
	}

	data.isVisible = testVisibility(dev_nodes, data);

	return closestNode->shader->shade(ray, data);
}

__global__ 
void renderScene(Color* dev_vfb, Camera* dev_cam, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	Ray ray = dev_cam->getScreenRay(x, y); 
	
	if (offset < RES_X * RES_Y)
	{
		dev_vfb[offset] = raytrace(ray, dev_geom, dev_shaders, dev_nodes);
	}
}

/**
 * Wrapper kernel function
*/
extern "C" 
void cudaRenderer(Color* dev_vfb, Camera* dev_cam, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes)
{
	initializeScene<<<1, 1>>>(dev_cam, dev_geom, dev_shaders, dev_nodes);

	//const int THREADS_PER_BLOCK = 32;  //32, 192, 64

	dim3 THREADS_PER_BLOCK(32, 32); // 32*32 = 1024 (max threads per block supported)

	dim3 BLOCKS(RES_X / THREADS_PER_BLOCK.x, RES_Y / THREADS_PER_BLOCK.y); 

	renderScene<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb, dev_cam, dev_geom, dev_shaders, dev_nodes);
}