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
#include "Phong.cuh"
#include "Refraction.cuh"
#include "Transform.cuh"
#include "Reflection.cuh"
#include "Layered.cuh"
#include "Fresnel.cuh"

__device__
bool needsAA[RES_X * RES_Y];

__device__ 
Camera* dev_cam;

__device__ 
Geometry* dev_geom[GEOM_MAX_SIZE];

__device__ 
Shader* dev_shaders[GEOM_MAX_SIZE];

__device__ 
Node* dev_nodes[GEOM_MAX_SIZE];

__device__
bool testVisibility(const Vector& from, const Vector& to)
{
	Ray ray;
	ray.start = from;
	ray.dir = to - from;
	ray.dir.normalize();

	IntersectionData temp;
	temp.dist = (to - from).length();

	for (int i = 0; i < GEOM_COUNT; i++)
	{
		if (dev_nodes[i]->intersect(ray, temp))
		{
			return false;
		}
	}

	return true;
}

__device__
Node* createNode(Geometry* geom, Shader* shader)
{
	if (GEOM_COUNT >= GEOM_MAX_SIZE)
	{
		return;
	}
	
	dev_geom[GEOM_COUNT]    = geom;
	dev_shaders[GEOM_COUNT] = shader;
	dev_nodes[GEOM_COUNT]   = new Node(dev_geom[GEOM_COUNT], dev_shaders[GEOM_COUNT]);

	return dev_nodes[GEOM_COUNT++];
}

__global__ 
void initializeScene()
{	
	dev_cam = new Camera;
	dev_cam->yaw = 0;
	dev_cam->pitch = 0;
	dev_cam->roll = 0;
	dev_cam->fov = 90;
	dev_cam->aspect = 4.0 / 3.0;
	dev_cam->pos = Vector(0, 120, -100);

	dev_cam->beginFrame();
	
	lightPos = Vector(0, 296, 100);
	lightColor = Color(1, 1, 1);
	lightPower = 60000;
	ambientLight = Color(0.2, 0.2, 0.2);

	createNode(new Plane(5), new OrenNayar(Color(0.0, 1.0, 0.0), 1.0));

	Layered* mirror = new Layered;
	mirror->addLayer(new Reflection(), Color(1, 1, 1), new Fresnel(10.0));

	Node* BackWall = createNode(new Plane(-300), new OrenNayar(Color(1.0, 1.0, 0.0), 1.0));
	BackWall->transform.rotate(0, 90, 0);
	
	Node* SideWallLeft = createNode(new Plane(-150), new OrenNayar(Color(1.0, 0.0, 0.0), 1.0));
	SideWallLeft->transform.rotate(0, 0, 90);

	Node* SideWallRight = createNode(new Plane(150), new OrenNayar(Color(0.0, 0.0, 1.0), 1.0));
	SideWallRight->transform.rotate(0, 0, 90);

	Node* Roof = createNode(new Plane(300), new OrenNayar(Color(0.96, 0.82, 0.46), 1.0));

	//createNode(new Sphere(Vector(-40, 50, 150), 30.0), new Phong(Color(0.5, 0.0, 0.5), 32));

	//createNode(new Sphere(Vector(60, 50, 120), 40.0), new Refraction(Color(0.9, 0.9, 0.9), 10));

	//createNode(new Sphere(Vector(0, 150, 150), 30.0), new Reflection(Color(0.9, 0.9, 0.9)));

	Layered* moreGlossy = new Layered;
	moreGlossy->addLayer(new Phong(Color(0.0, 0.0, 1.0), 32), Color(1.0, 1.0, 1.0)); 
	moreGlossy->addLayer(new Reflection(Color(1.0, 1.0, 1.0)), Color(1, 1, 1), new Fresnel(2.5));
	createNode(new Sphere(Vector(0, 50, 200), 40.0), moreGlossy);

	/*Layered* glossy = new Layered;
	glossy->addLayer(new Phong(Color(0.5, 0.0, 0.5), 32), Color(1.0, 1.0, 1.0)); 
	glossy->addLayer(new Reflection(Color(0.9, 0.9, 0.9)), Color(1, 1, 1), new Fresnel(3.0));
	createNode(new Sphere(Vector(-90, 50, 130), 40.0), glossy);*/

}

__device__ 
Color raytrace(Ray ray)
{
	IntersectionData data;
	Node* closestNode = nullptr;

	if (ray.depth > MAX_RAY_DEPTH)
	{
		return Color(0, 0, 0);
	}

	data.dist = 1e99;
	
	for (int i = 0; i < GEOM_COUNT; i++)
	{
		if (dev_nodes[i]->intersect(ray, data))
		{
			closestNode = dev_nodes[i];
		}
	}

	if (!closestNode)
	{
		return Color(0, 0, 0);
	}

	//Vector N = faceforward(ray.dir, data.normal);
	//data.normal = N;

	//data.isVisible = testVisibility(data);
	
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
void antiAliasing(Color* dev_vfb)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;	
		
	const int n_size = 5;
	Color neighs[n_size];
	neighs[0] = dev_vfb[offset];
	neighs[1] = dev_vfb[(x > 0 ? x - 1 : x) + y * blockDim.x * gridDim.x];
	neighs[2] = dev_vfb[(x + 1 < RES_X ? x + 1 : x) + y * blockDim.x * gridDim.x];
	neighs[3] = dev_vfb[x + (y > 0 ? y - 1 : y) * blockDim.x * gridDim.x];
	neighs[4] = dev_vfb[x + (y + 1 < RES_Y ? y + 1 : y) * blockDim.x * gridDim.x];

	Color average(0, 0, 0);
			
	for (int i = 0; i < n_size; ++i)
	{
		average += neighs[i];
	}
	average /= static_cast<float>(n_size);
			
	for (int i = 0; i < n_size; ++i)
	{
		if (tooDifferent(neighs[i], average))
		{
			needsAA[offset] = true;
			break;
		}
	}

	const double kernel[5][2] = {
			{ 0, 0 },
			{ 0.3, 0.3 },
			{ 0.6, 0 },
			{ 0, 0.6 },
			{ 0.6, 0.6 },
		};

	bool previewAA = false;

	if (previewAA)
	{
		if (needsAA[offset])
		{
			dev_vfb[offset] = Color(1, 0, 0);
		}
	}
	else
	{
		if (needsAA[offset])
		{
			Color result = dev_vfb[offset];
			
			for (int i = 1; i < n_size; ++i)
			{
				result += raytrace(dev_cam->getScreenRay(x + kernel[i][0], y + kernel[i][1]));
			}
			dev_vfb[offset] = result / static_cast<float>(n_size);
		}
	}
}

__global__ 
void renderScene(Color* dev_vfb)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	
	if (offset < RES_X * RES_Y)
	{
		dev_vfb[offset] = raytrace(dev_cam->getScreenRay(x, y));
	}
}

__global__
void freeMemory()
{
	delete dev_cam;
	delete [] dev_geom;
	delete [] dev_nodes;
	delete [] dev_shaders;
}

/**
 * Wrapper kernel function
*/
extern "C" 
void cudaRenderer(Color* dev_vfb)
{
	initializeScene<<<1, 1>>>();

	dim3 THREADS_PER_BLOCK(32, 32); // 32*32 = 1024 (max threads per block supported)
	dim3 BLOCKS(RES_X / THREADS_PER_BLOCK.x, RES_Y / THREADS_PER_BLOCK.y); 
	
	// first pass
	renderScene<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb);

#ifdef ANTI_ALIASING

	//second pass
	antiAliasing<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb);

#endif
	
}

extern "C"
void freeDeviceMemory()
{
	freeMemory<<<1, 1>>>();
}