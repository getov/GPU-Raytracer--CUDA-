//#ifndef __CUDACC__  
//    #define __CUDACC__
//#endif
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
#include "OrenNayar.cuh"

__device__
bool needsAA[RES_X * RES_Y];

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

__device__
void createNode(Geometry* geom, Shader* shader,
				Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes)
{
	if (GEOM_COUNT >= GEOM_MAX_SIZE)
	{
		return;
	}

	dev_geom[GEOM_COUNT]    = geom;
	dev_shaders[GEOM_COUNT] = shader;
	dev_nodes[GEOM_COUNT]   = new Node(dev_geom[GEOM_COUNT], dev_shaders[GEOM_COUNT]);
	++GEOM_COUNT;
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

	createNode(new Plane(5), new OrenNayar(Color(0.0, 1.0, 0.0), 1.0),
			   dev_geom, dev_shaders, dev_nodes);

	createNode(new Sphere(Vector(-150, 40, 180), 20.0), new Lambert(Color(1.0, 1.0, 0.0)),
			   dev_geom, dev_shaders, dev_nodes);

	createNode(new Sphere(Vector(-50, 40, 180), 20.0), new Lambert(Color(0.5, 0.5, 0.5)),
			   dev_geom, dev_shaders, dev_nodes);

	createNode(new Sphere(Vector(-100, 40, 180), 20.0), new OrenNayar(Color(0.5, 0.5, 0.5), 1.0),
			   dev_geom, dev_shaders, dev_nodes);

	createNode(new Sphere(Vector(0, 40, 180), 20.0), new OrenNayar(Color(0.5, 0.5, 0.5), 0.5),
			   dev_geom, dev_shaders, dev_nodes);

	createNode(new Sphere(Vector(-100, 40, 220), 20.0), new OrenNayar(Color(0.5, 0.5, 0.5), 0.5),
			   dev_geom, dev_shaders, dev_nodes);

	createNode(new Sphere(Vector(-50, 40, 220), 20.0), new OrenNayar(Color(0.0, 0.5, 0.5), 0.2),
			   dev_geom, dev_shaders, dev_nodes);
	
	createNode(new Sphere(Vector(0, 40, 220), 20.0), new OrenNayar(Color(0.0, 0.0, 0.5), 0.9),
			   dev_geom, dev_shaders, dev_nodes);

	createNode(new Sphere(Vector(80, 40, 220), 20.0), new OrenNayar(Color(0.5, 0.0, 0.5), 0.9),
			   dev_geom, dev_shaders, dev_nodes);


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

/**
 * The function checks if one of the red, green or blue components
 * of the colors a and b are too different.
 * @return true - if the difference is bigger than the THRESHOLD
 * @return false - if the difference is lower than the THRESHOLD
*/
__device__
bool tooDifferent(const Color& a, const Color& b)
{
	const float THRESHOLD = 0.1;
	return (fabs(a.r - b.r) > THRESHOLD ||
		     fabs(a.g - b.g) > THRESHOLD ||
		     fabs(a.b - b.b) > THRESHOLD);
}

__global__ 
void renderScene(Color* dev_vfb, Camera* dev_cam, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	if (offset < RES_X * RES_Y)
	{
		// first pass
		dev_vfb[offset] = raytrace(dev_cam->getScreenRay(x, y), dev_geom, dev_shaders, dev_nodes);
	
#ifdef ANTI_ALIASING
		const double kernel[5][2] = {
					{ 0, 0 },
					{ 0.3, 0.3 },
					{ 0.6, 0 },
					{ 0, 0.6 },
					{ 0.6, 0.6 },
				};
		
		// second pass
		Color neighs[5];
		neighs[0] = dev_vfb[offset];
		neighs[1] = dev_vfb[(x > 0 ? x - 1 : x) + y * blockDim.x * gridDim.x];
		neighs[2] = dev_vfb[(x + 1 < RES_X ? x + 1 : x) + y * blockDim.x * gridDim.x];
		neighs[3] = dev_vfb[x + (y > 0 ? y - 1 : y) * blockDim.x * gridDim.x];
		neighs[4] = dev_vfb[x + (y + 1 < RES_Y ? y + 1 : y) * blockDim.x * gridDim.x];

		Color average(0, 0, 0);
			
		for (int i = 0; i < 5; ++i)
		{
			average += neighs[i];
		}
		average /= 5.0f;
			
		for (int i = 0; i < 5; ++i)
		{
			if (tooDifferent(neighs[i], average))
			{
				needsAA[offset] = true;
				break;
			}
		}

		if (needsAA[offset])
		{
			Color result = dev_vfb[offset];
			
			for (int i = 1; i < 5; ++i)
			{
				result += raytrace(dev_cam->getScreenRay(x + kernel[i][0], y + kernel[i][1]), dev_geom, dev_shaders, dev_nodes);
			}
			dev_vfb[offset] = result / 5.0f;
		}
#endif
	}
}

/**
 * Wrapper kernel function
*/
extern "C" 
void cudaRenderer(Color* dev_vfb, Camera* dev_cam, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes)
{
	initializeScene<<<1, 1>>>(dev_cam, dev_geom, dev_shaders, dev_nodes);

	dim3 THREADS_PER_BLOCK(32, 32); // 32*32 = 1024 (max threads per block supported)

	dim3 BLOCKS(RES_X / THREADS_PER_BLOCK.x, RES_Y / THREADS_PER_BLOCK.y); 

	renderScene<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb, dev_cam, dev_geom, dev_shaders, dev_nodes);
}