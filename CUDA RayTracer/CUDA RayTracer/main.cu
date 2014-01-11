#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SDL/SDL.h>
#include <iostream>
#include "cuda_renderer.cuh"
#include "Vector3D.cuh"
#include <vector>
#include "sdl.cuh"
#include "Color.cuh"
#include "Camera.cuh"
#include "Matrix.cuh"
#include "IGeometry.cuh"
#include "IShader.cuh"
#include "Node.cuh"
#include "Lambert.cuh"
#include "Plane.cuh"

using namespace std;

extern "C" void cudaRenderer(Color* dev_vfb, Camera* dev_cam, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes);

Color vfb[RES_X][RES_Y];

// used for GPU operations
Color vfb_linear[RES_X * RES_Y];

const int ARR_SIZE = 3;

Camera* camera;
Geometry* geometry[ARR_SIZE];
Shader* shaders[ARR_SIZE];
Node* nodes[ARR_SIZE];

void cudaStartTimer(cudaEvent_t& start, cudaEvent_t& stop)
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void cudaStopTimer(cudaEvent_t& start, cudaEvent_t& stop)
{
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float  elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to render:  %3.1f ms\n", elapsedTime);
	
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void convertDeviceToHostBuffer()
{
	for (int i = 0; i < RES_X; ++i)
	{
		for (int j = 0; j < RES_Y; ++j)
		{
			vfb[i][j] = vfb_linear[i * RES_X + j];
		}
	}
}

int main(int argc, char** argv)
{
	if (!initGraphics(RES_X, RES_Y))
	{
		return -1;
	}

	// capture the start time
	cudaEvent_t start, stop;
	cudaStartTimer(start, stop);

	// 1. allocate memory for vfb on the GPU
	Color* dev_vfb;
	cudaMalloc((void**)&dev_vfb, sizeof(Color) * RES_X * RES_Y);

	Camera* dev_cam;
	cudaMalloc((void**)&dev_cam, sizeof(Camera));

	Geometry** dev_geom;
	cudaMalloc((void**)&dev_geom, sizeof(Geometry) * ARR_SIZE);

	Shader** dev_shaders;
	cudaMalloc((void**)&dev_shaders, sizeof(Shader) * ARR_SIZE);

	Node** dev_nodes;
	cudaMalloc((void**)&dev_nodes, sizeof(Node) * ARR_SIZE);
	
	// 2. memcpy HostToDevice
	cudaMemcpy(dev_vfb, vfb_linear, sizeof(Color) * RES_X * RES_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cam, camera, sizeof(Camera), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_geom, geometry, sizeof(Geometry) * ARR_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_shaders, shaders, sizeof(Shader) * ARR_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nodes, nodes, sizeof(Node) * ARR_SIZE, cudaMemcpyHostToDevice);

	// 3. call kernels
	// - InitializeScene
	// - RenderScene
	cudaRenderer(dev_vfb, dev_cam, dev_geom, dev_shaders, dev_nodes);

	// 4. memcpy DeviceToHost
	cudaMemcpy(vfb_linear, dev_vfb, sizeof(Color) * RES_X * RES_Y, cudaMemcpyDeviceToHost);
	/*cudaMemcpy(camera, dev_cam, sizeof(Camera), cudaMemcpyDeviceToHost);
	cudaMemcpy(geometry, dev_geom, sizeof(Geometry), cudaMemcpyDeviceToHost);
	cudaMemcpy(shaders, dev_shaders, sizeof(Shader), cudaMemcpyDeviceToHost);
	cudaMemcpy(nodes, dev_nodes, sizeof(Node), cudaMemcpyDeviceToHost);*/

	// get stop time, and display the timing results
	cudaStopTimer(start, stop);

	// 5. free memory
	cudaFree(dev_vfb);
	cudaFree(dev_cam);
	cudaFree(dev_geom);
	cudaFree(dev_shaders);
	cudaFree(dev_nodes);

	// convert the linear array to our 2D array
	convertDeviceToHostBuffer();
	
	displayVFB(vfb);

	waitForUserExit();
	
	closeGraphics();

	return EXIT_SUCCESS;
}