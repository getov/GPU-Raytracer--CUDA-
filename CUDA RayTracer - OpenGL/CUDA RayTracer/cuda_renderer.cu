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
#include "CameraController.cuh"
#include "RaytracerControls.cuh"
#include "Settings.cuh"
#include "WaterWaves.cuh"
#include "Scene.cuh"

__device__
bool needsAA[VFB_MAX_SIZE * VFB_MAX_SIZE];

__device__
Color colorBuffer[VFB_MAX_SIZE * VFB_MAX_SIZE];
 
__device__
CameraController* controller;

__device__ Scene* scene;

__device__
bool testVisibility(const Vector& from, const Vector& to)
{
	Ray ray;
	ray.start = from;
	ray.dir = to - from;
	ray.dir.normalize();

	IntersectionData temp;
	temp.dist = (to - from).length();

	for (int i = 0; i < scene->dev_nodes.size(); ++i)
	{
		if (scene->dev_nodes[i]->intersect(ray, temp))
		{
			return false;
		}
	}

	return true;
}

__device__
Node* createNode(Geometry* geom, Shader* shader, Texture* tex = nullptr)
{
	scene->dev_geom.push_back(geom);
	scene->dev_shaders.push_back(shader);
	scene->dev_textures.push_back(tex);

	Node* node = new Node(geom, shader, tex);
	scene->dev_nodes.push_back(node);

	return node;
}

__global__ 
void initializeScene(short sceneID, int RES_X, int RES_Y)
{	
	precomputeColorCache();

	scene = new Scene;

	scene->dev_cam = new Camera;
	scene->dev_cam->yaw = 0;
	scene->dev_cam->pitch = 0;
	scene->dev_cam->roll = 0;
	scene->dev_cam->fov = 90;
	scene->dev_cam->aspect = static_cast<float>(RES_X) / RES_Y;
	scene->dev_cam->pos = Vector(0, 150, -100);
	scene->dev_cam->beginFrame();

	controller = new CameraController(*(scene->dev_cam), 10.f);
	
	switch (sceneID)
	{
		case CORNELL_BOX:
		{
			scene->dev_lights.push_back(new RectLight(Vector(0, 296, 200), Vector(0, 0, 0), Vector(50, 34, 34), Color(1, 1, 1), 20, 6, 6));
			//scene->dev_lights.push_back(new RectLight(Vector(-70, 296, 200), Vector(0, 0, 0), Vector(50, 34, 34), Color(0, 0.5, 0.5), 20, 6, 6));
			//scene->dev_lights.push_back(new SpotLight(Vector(0, 296, 180), Vector(0, -1, 1), Color(1, 1, 1), 60, 15.0, 35.0));
			//scene->dev_lights.push_back(new PointLight(Vector(0, 296, 200), Color(1, 1, 1), 50000));

			createNode(new Plane(5, 300, 300), new Lambert(Color(0xF5E08C)));

			Layered* mirror = new Layered;
			mirror->addLayer(new Reflection(), Color(1, 1, 1), new Fresnel(10.0));

			Node* BackWall = createNode(new Plane(-300, 300, 300), new Lambert(Color(0xF5E08C)));
			BackWall->transform.rotate(0, 90, 0);
	
			Node* SideWallLeft = createNode(new Plane(-150, 300, 300), new Lambert(Color(1.0, 0.0, 0.0)));
			SideWallLeft->transform.rotate(0, 0, 90);

			Node* SideWallRight = createNode(new Plane(150, 300, 300), new Lambert(Color(0.0, 0.0, 1.0)));
			SideWallRight->transform.rotate(0, 0, 90);

			Node* Roof = createNode(new Plane(300, 300, 300), new Lambert(Color(0xF5E08C)));

			Layered* moreGlossy = new Layered;
			moreGlossy->addLayer(new Phong(Color(0.0, 0.0, 1.0), 32), Color(1.0, 1.0, 1.0)); 
			moreGlossy->addLayer(new Reflection(Color(1.0, 1.0, 1.0)), Color(1, 1, 1), new Fresnel(2.5));
			createNode(new Sphere(Vector(0, 50, 200), 40.0), moreGlossy);

			//createNode(new Sphere(Vector(0, 50, 200), 40.0), new OrenNayar(Color(0.0, 0.0, 1.0), 1.0));

			Node* rectMirror = createNode(new Plane(0, 60, 80), mirror);
			rectMirror->transform.rotate(0, 90, 0);
			rectMirror->transform.translate(Vector(0, 120, 298));

			break;
		}
		case ROAMING:
		{
			scene->dev_lights.push_back(new PointLight(Vector(0, 296, 200), Color(1, 1, 1), 50000));

			createNode(new Plane(5), new Lambert(Color(0.5, 0.5, 0.5)));
			//createNode(new Plane(500), new OrenNayar(Color(0.5, 0.5, 0.5), 1.0));
			createNode(new Sphere(Vector(0, 50, 200), 40.0), new Phong(Color(0, 0, 1), 32));

			break;
		}
		case SEA:
		{
			scene->dev_lights.push_back(new PointLight(Vector(0, 300, -100), Color(0.2, 0.2, 0), 500000));

			createNode(new Plane(-300, 1000, 1000), new Lambert(Color(0x0AB6FF)));  // 0.1448, 0.4742, 0.6804   0x0AB6FF
			Layered* water = new Layered;
			water->addLayer(new Refraction(Color(0.9, 0.9, 0.9), 1.33), Color(1.0, 1.0, 1.0));
			water->addLayer(new Reflection(Color(0.9, 0.9, 0.9)), Color(1.0, 1.0, 1.0), new Fresnel(1.33));
	
			Node* waterGeom = createNode(new Plane(0, 100, 100), water, new WaterWaves(0.2));
			waterGeom->transform.scale(10, 1, 10);

			Node* island = createNode(new Sphere(Vector(0, 0, 0), 100.0), new Lambert(Color(0, 1, 0)));
			island->transform.scale(10, 2, 15);
			island->transform.translate(Vector(10, -20, 1500));

			break;
		}
		default:
			break;
	}
}

__global__
void update(float elapsedTime, float currentTime)
{
	scene->waves = currentTime;
}

extern "C"
void updateScene(float elapsedTime, float currentTime)
{
	update<<<1, 1>>>(elapsedTime, currentTime);
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

	for (int i = 0; i < scene->dev_nodes.size(); ++i)
	{
		if (scene->dev_nodes[i]->intersect(ray, data))
		{
			closestNode = scene->dev_nodes[i];
		}
	}

	// check if the closest intersection point is actually a light:
	bool hitLight = false;
	Color hitLightColor;
	for (int i = 0; i < scene->dev_lights.size(); ++i)
	{
		if (scene->dev_lights[i]->intersect(ray, data.dist))
		{
			hitLight = true;
			hitLightColor = scene->dev_lights[i]->getColor();
		}
	}
	if (hitLight) return hitLightColor;

	if (!closestNode)
	{
		//return Color(0, 0, 0);
		return Color(0.55f, 0.8f, 0.95f); // skyblue
		//return Color(1, 1, 1);
	}

	if (closestNode->bumpTex != nullptr)
	{
		closestNode->bumpTex->modifyNormal(data);
	}
	
	return closestNode->shader->shade(ray, data);
}

/**
 * @brief - The function checks if one of the red, green or blue components
 * of the colors a and b are too different.
 * @return true - if the difference is bigger than the THRESHOLD
 * @return false - if the difference is lower than the THRESHOLD
*/
__device__
inline bool tooDifferent(const Color& a, const Color& b)
{
	const float THRESHOLD = 0.1; // max color threshold; if met on any of the three channels, consider the colors too different
	for (int comp = 0; comp < 3; comp++) {
		float theMax = dev_max(a[comp], b[comp]);
		float theMin = dev_min(a[comp], b[comp]);

		// compare a single channel of the two colors. If the difference between them is large,
		// but they aren't overexposed, the difference will be visible: needs anti-aliasing.
		if (theMax - theMin > THRESHOLD && theMin < 1.33f) 
			return true;
	}
	return false;
}

__global__
void toGrayscale(uchar4* dev_vfb, bool previewAA, int RES_X, int RES_Y)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	dev_vfb[offset].x = convertTo8bit_sRGB_cached(colorBuffer[offset].intensityPerceptual());
	dev_vfb[offset].y = convertTo8bit_sRGB_cached(colorBuffer[offset].intensityPerceptual());
	dev_vfb[offset].z = convertTo8bit_sRGB_cached(colorBuffer[offset].intensityPerceptual());
}

__global__
void blurScene(uchar4* dev_vfb, bool previewAA, int RES_X, int RES_Y)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	Color result = colorBuffer[offset];

	// take just straight up-down and right-left neighbours
	result += colorBuffer[(x > 0 ? x - 1 : x) + y * blockDim.x * gridDim.x] +
			  colorBuffer[(x + 1 < RES_X ? x + 1 : x) + y * blockDim.x * gridDim.x] +
			  colorBuffer[x + (y > 0 ? y - 1 : y) * blockDim.x * gridDim.x] +
			  colorBuffer[x + (y + 1 < RES_Y ? y + 1 : y) * blockDim.x * gridDim.x];

	colorBuffer[offset] = result / 5.0f;

	// take all neighbours (up-down, right-left and diagonals)
	//result += colorBuffer[(x > 0 ? x - 1 : x) + y * blockDim.x * gridDim.x] +
	//		    colorBuffer[(x > 0 ? x - 1 : x) + (y > 0 ? y - 1 : y) * blockDim.x * gridDim.x] +
	//		    colorBuffer[(x + 1 < RES_X ? x + 1 : x) + y * blockDim.x * gridDim.x] +
	//		    colorBuffer[(x + 1 < RES_X ? x + 1 : x) + (y + 1 < RES_Y ? y + 1 : y) * blockDim.x * gridDim.x] +
	//		    colorBuffer[x + (y > 0 ? y - 1 : y) * blockDim.x * gridDim.x] +
	//		    colorBuffer[(x > 0 ? x - 1 : x) + (y > 0 ? y - 1 : y) * blockDim.x * gridDim.x] +
	//		    colorBuffer[x + (y + 1 < RES_Y ? y + 1 : y) * blockDim.x * gridDim.x] +
	//		    colorBuffer[(x < 0 ? x + 1 : x) + (y + 1 < RES_Y ? y + 1 : y) * blockDim.x * gridDim.x];
	//
	//colorBuffer[offset] = result / static_cast<float>(9.0);

	dev_vfb[offset].x = convertTo8bit_sRGB_cached(colorBuffer[offset].r);
	dev_vfb[offset].y = convertTo8bit_sRGB_cached(colorBuffer[offset].g);
	dev_vfb[offset].z = convertTo8bit_sRGB_cached(colorBuffer[offset].b);
}

__global__
void antiAliasing(uchar4* dev_vfb, bool previewAA, int RES_X, int RES_Y)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;	
		
	const int n_size = 5;
	Color neighs[n_size];
	neighs[0] = colorBuffer[offset];
	neighs[1] = colorBuffer[(x > 0 ? x - 1 : x) + y * blockDim.x * gridDim.x];
	neighs[2] = colorBuffer[(x + 1 < RES_X ? x + 1 : x) + y * blockDim.x * gridDim.x];
	neighs[3] = colorBuffer[x + (y > 0 ? y - 1 : y) * blockDim.x * gridDim.x];
	neighs[4] = colorBuffer[x + (y + 1 < RES_Y ? y + 1 : y) * blockDim.x * gridDim.x];
	
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

	if (previewAA)
	{
		if (needsAA[offset])
		{
			dev_vfb[offset].x = 255;
			dev_vfb[offset].y = 0;
			dev_vfb[offset].z = 0;
		}
	}
	else
	{
		if (needsAA[offset])
		{
			Color result = colorBuffer[offset];
			
			for (int i = 1; i < n_size; ++i)
			{
				result += raytrace(scene->dev_cam->getScreenRay(x + kernel[i][0], y + kernel[i][1], RES_X, RES_Y));
			}
			colorBuffer[offset] = result / static_cast<float>(n_size);
			dev_vfb[offset].x = convertTo8bit_sRGB_cached(colorBuffer[offset].r);
			dev_vfb[offset].y = convertTo8bit_sRGB_cached(colorBuffer[offset].g);
			dev_vfb[offset].z = convertTo8bit_sRGB_cached(colorBuffer[offset].b);
		}
	}
}

__global__ 
void renderScene(uchar4* dev_vfb, int RES_X, int RES_Y)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (offset < RES_X * RES_Y)
	{
		colorBuffer[offset] = raytrace(scene->dev_cam->getScreenRay(x, y, RES_X, RES_Y));
		dev_vfb[offset].x = convertTo8bit_sRGB_cached(colorBuffer[offset].r);
		dev_vfb[offset].y = convertTo8bit_sRGB_cached(colorBuffer[offset].g);
		dev_vfb[offset].z = convertTo8bit_sRGB_cached(colorBuffer[offset].b);
	}
}

__global__
void freeMemory()
{
	//delete dev_cam;
	delete controller;
	delete scene;

	printf("asd\n");
}

/**
 * Wrapper kernel function
*/
extern "C"
void initScene()
{
	initializeScene<<<1, 1>>>(GlobalSettings::sceneID, GlobalSettings::RES_X, GlobalSettings::RES_Y);
}

__global__
void camBeginFrame()
{
	scene->dev_cam->beginFrame();
}

extern "C"
void cameraBeginFrame()
{
	camBeginFrame<<<1, 1>>>();
}

extern "C" 
void cudaRenderer(uchar4* dev_vfb)
{
	dim3 THREADS_PER_BLOCK(8, 8); // 8*8 - most optimal; 32*32 = 1024 (max threads per block supported)
	dim3 BLOCKS(GlobalSettings::RES_X / THREADS_PER_BLOCK.x, GlobalSettings::RES_Y / THREADS_PER_BLOCK.y); 

	// first pass
	renderScene<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb, GlobalSettings::RES_X, GlobalSettings::RES_Y);

	if (GlobalSettings::AAEnabled)
	{
		//second pass
		antiAliasing<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb, GlobalSettings::previewAA, GlobalSettings::RES_X, GlobalSettings::RES_Y);
	}

	if (GlobalSettings::blur)
	{
		blurScene<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb, GlobalSettings::previewAA, GlobalSettings::RES_X, GlobalSettings::RES_Y);
	}

	if (GlobalSettings::grayscale)
	{
		toGrayscale<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_vfb, GlobalSettings::previewAA, GlobalSettings::RES_X, GlobalSettings::RES_Y);
	}
}

extern "C"
void freeDeviceMemory()
{	
	freeMemory<<<1, 1>>>();
}