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
#include "Sphere.cuh"
	
using namespace std;

extern "C" void cudaRenderer(Color* dev_vfb, Camera* dev_cam, Geometry** dev_geom, Shader** dev_shaders, Node** dev_nodes);

// virtual framebuffer
Color vfb[RES_Y][RES_X];

// virtual framebuffer used for GPU operations
Color vfb_linear[RES_X * RES_Y]; 

Camera* camera;
Geometry* geometry[GEOM_MAX_SIZE];
Shader* shaders[GEOM_MAX_SIZE];
Node* nodes[GEOM_MAX_SIZE];

/**
 * Function that prints CUDA specs of the 
 * of the GPU device/s on the console
*/
void printGPUSpecs()
{
	cudaDeviceProp  prop;
    int count;
    cudaGetDeviceCount(&count);
	
    for (int i = 0; i < count; ++i) 
	{
        cudaGetDeviceProperties( &prop, i );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap:  " );

        if (prop.deviceOverlap)
		{
            printf( "Enabled\n" );
		}
        else
		{
            printf( "Disabled\n");
		}
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
		{
            printf( "Enabled\n" );
		}
        else
		{
            printf( "Disabled\n" );
		}

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}

/**
 * Wrapper function that creates timer and captures the start and stop time
 * @param start - output - captures the start time
 * @param stop - output - captires the stop time
*/
void cudaStartTimer(cudaEvent_t& start, cudaEvent_t& stop)
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

/**
 * Wrapper function that takes the previously captured start and stop time
 * from cudaStartTimer() function and calculates the elapsed time
 * @param start - the start time that is previously captured by cudaStartTimer()
 * @param stop - the stop time that is previously captured by cudaStartTimer()
 * @reference - cudaStartTimer(cudaEvent_t& start, cudaEvent_t& stop)
*/
void cudaStopTimer(cudaEvent_t& start, cudaEvent_t& stop)
{
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float  elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf( "Time to render:  %3.1f ms\n\n", elapsedTime);
	
	cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * function that converts the linear array vfb_linear
 * into the 2D array vfb
 *
 * This is needed because we pass linear array to the GPU
 * to process our pixel data and then we convert it to 
 * 2D array in order to display our pixel data with SDL
*/
void convertDeviceToHostBuffer()
{
	for (int i = 0; i < RES_Y; ++i)
	{
		for (int j = 0; j < RES_X; ++j)
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
	cudaMalloc((void**)&dev_geom, sizeof(Geometry*) * GEOM_MAX_SIZE);

	Shader** dev_shaders;
	cudaMalloc((void**)&dev_shaders, sizeof(Shader*) * GEOM_MAX_SIZE);

	Node** dev_nodes;
	cudaMalloc((void**)&dev_nodes, sizeof(Node*) * GEOM_MAX_SIZE);
	
	// 2. memcpy HostToDevice
	cudaMemcpy(dev_vfb, vfb_linear, sizeof(Color) * RES_X * RES_Y, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cam, camera, sizeof(Camera), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_geom, geometry, sizeof(Geometry*) * GEOM_MAX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_shaders, shaders, sizeof(Shader*) * GEOM_MAX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nodes, nodes, sizeof(Node*) * GEOM_MAX_SIZE, cudaMemcpyHostToDevice);

	// 3. call kernels
	// - InitializeScene
	// - RenderScene
	cudaRenderer(dev_vfb, dev_cam, dev_geom, dev_shaders, dev_nodes);

	// 4. memcpy DeviceToHost
	cudaMemcpy(vfb_linear, dev_vfb, sizeof(Color) * RES_X * RES_Y, cudaMemcpyDeviceToHost);

	// get stop time, and display the timing results
	cudaStopTimer(start, stop);

	printGPUSpecs();

	// 5. free memory
	cudaFree(dev_vfb);
	cudaFree(dev_cam);
	cudaFree(dev_geom);
	cudaFree(dev_shaders);
	cudaFree(dev_nodes);

	convertDeviceToHostBuffer();

	displayVFB(vfb);

	handleUserInput();
	
	closeGraphics();

	return EXIT_SUCCESS;
}