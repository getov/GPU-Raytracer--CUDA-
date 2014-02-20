#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "Color.cuh"
#include "Vector3D.cuh"
#include "Settings.cuh"
#include "IGeometry.cuh"
#include "Camera.cuh"
#include "IShader.cuh"
#include "Node.cuh"
#include "custom_vector.cuh"
#include "CameraController.cuh"
#include "Scene.cuh"

class Scene;

extern __device__ Scene* scene;

/**
 * @brief - Function that performs the raytracing algorithm.
 * @param ray - the ray 'shot' from the camera towards the screen
 * @return - returns the color for the pixel on which is performed the algorithm
*/
__device__ Color raytrace(Ray ray);

/**
 * @brief - Function that checks if the point 'to' is visible from the point 'from'
 * @param from - point that is near a surface
 * @param to   - point that is near a light
 * @return true - if the poins 'from' and 'to' are visible from each other, otherwise 
 * @return false - otherwise
*/
__device__ bool testVisibility(const Vector& from, const Vector& to);

/**
 * @brief - Helper function to create a Node of geometry, shader and texture
 * @param geom   - The geometry to be added to the Node
 * @param shader - The shader to be added to the Node
 * @param tex    - The texture to be added to the Node
*/
__device__ Node* createNode(Geometry* geom, Shader* shader, Texture* tex = nullptr);

/**
 * @brief - Setup and precompute different arrangements of scenes to render
 * @param sceneID - The ID of the scene to be rendered
 * @param RES_X - the horizontal window resolution
 * @param RES_Y - the vertical window resolution
*/
__global__ void initializeScene(short sceneID, int RES_X, int RES_Y);

/**
 * @brief - update scene elements when Real-Time rendering is enabled
 * @param elapsedTime - The difference between the time before the updates
 * take place and the time after the updates (in seconds)
 * @param currentTime - The current time since the beginning of the loop (in seconds)
*/
__global__ void update(double elapsedTime, double currentTime);

/**
 * @brief - Convert the scene into Grayscale image
 * @param dev_vfb - the pixel buffer that holds data of 
 * the colors for each pixel on the screen.
 * @param RES_X - the horizontal window resolution
 * @param RES_Y - the vertical window resolution
*/
__global__ void toGrayscale(uchar4* dev_vfb, int RES_X, int RES_Y);

/**
 * @brief - Blur the image (full scene blur)
 * @param dev_vfb - the pixel buffer that holds data of 
 * the colors for each pixel on the screen.
 * @param RES_X - the horizontal window resolution
 * @param RES_Y - the vertical window resolution
*/
__global__ void blurScene(uchar4* dev_vfb, int RES_X, int RES_Y);

/**
 * @brief - Perform adaptive Anti-Aliasing
 * @param dev_vfb - the pixel buffer that holds data of 
 * the colors for each pixel on the screen.
 * @param previewAA - if it is 'true', it shows the pixels that need
 * Anti-Aliasing in red color.
 * @param RES_X - the horizontal window resolution
 * @param RES_Y - the vertical window resolution
*/
__global__ void antiAliasing(uchar4* dev_vfb, bool previewAA, int RES_X, int RES_Y);

/**
 * @brief - Perform the raytracing algorithm for each pixel on the screen and 
 * fill the pixel buffer data. 
 * @param dev_vfb - the pixel buffer that holds data of 
 * the colors for each pixel on the screen.
 * @param RES_X - the horizontal window resolution
 * @param RES_Y - the vertical window resolution
 * @reference - __device__ Color raytrace(Ray ray)
*/
__global__ void renderScene(uchar4* dev_vfb, int RES_X, int RES_Y);

/**
 * @brief - Free the allocated memory
*/
__global__ void freeMemory();

/**
 * @brief - execute Camera::beginFrame()
*/
__global__ void camBeginFrame();

/**
 * @brief - Wrapper functions that are used to launch the kernels
*/
extern "C"
{
	/**
	 * @brief - Wrapper function for the update<<< , , >>>() kernel
	 * @param elapsedTime - The difference between the time before the updates
	 * take place and the time after the updates (in seconds)
	 * @param currentTime - The current time since the beginning of the loop (in seconds)
	 * @reference - __global__ void update(double elapsedTime, double currentTime);
	*/
	void updateScene(const double& elapsedTime, const double& currentTime);

	/**
	 * @brief - Wrapper function for the initializeScene<<< , , >>>() kernel
	 * @reference - __global__ void initializeScene(short sceneID, int RES_X, int RES_Y);
	*/
	void initScene();

	/**
	 * @brief - Wrapper function for the camBeginFrame<<< , , >>>() kernel
	 * @reference - __global__ void camBeginFrame();
	*/
	void cameraBeginFrame();

	/**
	 * @brief - Wrapper function for the kernels:
	 * renderScene<<< , , >>>()
	 * antiAliasing<<< , , >>>()
	 * blurScene<<< , , >>>()
	 * toGrayscale<<< , , >>>()
	 * Each of the kernels is launched (if marked for executing)
	 * with 8x8 = 64 threads per block and (REX_X / 8) * (RES_Y / 8) blocks
	 * @reference - __global__ void renderScene(uchar4* dev_vfb, int RES_X, int RES_Y)
	 *			  - __global__ void antiAliasing(uchar4* dev_vfb, bool previewAA, int RES_X, int RES_Y)
	 *            - __global__ void blurScene(uchar4* dev_vfb, int RES_X, int RES_Y)
	 *            - __global__ void toGrayscale(uchar4* dev_vfb, int RES_X, int RES_Y)
	*/
	void cudaRenderer(uchar4* dev_vfb);

	/**
	 * @brief - Wrapper function for the freeMemory<<< , , >>>() kernel
	 * @reference - __global__ void freeMemory();
	*/
	void freeDeviceMemory();
}

#endif