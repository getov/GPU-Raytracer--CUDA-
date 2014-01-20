#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <GL\glfw.h>

#include <SDL/SDL.h>
#include <iostream>
#include <string>
#include <sstream>
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

extern "C" void cudaRenderer(Color* dev_vfb);
extern "C" void freeDeviceMemory();
extern "C" void initScene();

extern "C" void updateScene();

unsigned frameCount;
unsigned lastFrameEnd;
unsigned lastTitleUpdateTime;
unsigned lastTitleUpdateFrameCount;
const char* const appName = "CUDA Traycer";

// virtual framebuffer
Color vfb[RES_Y][RES_X];

// virtual framebuffer used for GPU operations
Color vfb_linear[RES_X * RES_Y]; 

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
 * from cudaStartTimer() function, calculates the elapsed time,
 * prints it on the console and shows it on the window frame
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
	
	char info[128];
	sprintf(info, "Time to render: %3.1f ms", elapsedTime);
	SDL_WM_SetCaption(info, NULL);

	cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void displayFrameCounter()
{
	++frameCount;

	const unsigned now = SDL_GetTicks();
	const unsigned frameTime = now - lastFrameEnd;
	const unsigned titleUpdateTimeDelta = now - lastTitleUpdateTime;

	if (titleUpdateTimeDelta > 1000)
	{
		const unsigned framesDelta = frameCount - lastTitleUpdateFrameCount;
		const unsigned meanFrameTime = titleUpdateTimeDelta / framesDelta;
		const unsigned fps = framesDelta * 1000 / titleUpdateTimeDelta;

		std::ostringstream title;
		title << appName << "\t\t\t mean frame time: " << meanFrameTime << " ms || fps: " << fps;
		title.flush();

		SDL_WM_SetCaption(title.str().c_str(), NULL);

		lastTitleUpdateTime = now;
		lastTitleUpdateFrameCount = frameCount;
	}

	lastFrameEnd = SDL_GetTicks();
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

	printGPUSpecs();

	cudaDeviceSetLimit(cudaLimitStackSize, STACK_SIZE);
	
	/*size_t stackLimit;
	cudaDeviceGetLimit(&stackLimit, cudaLimitMallocHeapSize);
	printf("%d\n", stackLimit);*/

	// allocate memory for vfb on the GPU
	Color* dev_vfb;
	cudaMalloc((void**)&dev_vfb, sizeof(Color) * RES_X * RES_Y);
	
	// memcpy HostToDevice
	cudaMemcpy(dev_vfb, vfb_linear, sizeof(Color) * RES_X * RES_Y, cudaMemcpyHostToDevice);

#ifdef REAL_TIME_RENDERING

	initScene();
	
	Uint32 lastTime = SDL_GetTicks();
	bool running = true;
	while (running)
	{
		Uint32 thisTime = SDL_GetTicks();

		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			switch (event.type)
			{
				case SDL_KEYDOWN:
				{
					switch (event.key.keysym.sym)
					{
						case SDLK_ESCAPE:
									running = false;
									break;
						case SDLK_UP:
									printf("SDLK_UP\n");
									updateScene();
									break;
						case SDLK_DOWN:
									printf("SDLK_DOWN\n");
									updateScene();
									break;
						default:
							break;
					}
				}
				case SDL_KEYUP:
				{
					switch (event.key.keysym.sym)
					{
						case SDLK_UP:
									//forw = false;
									break;
						default:
							break;
					}
				}
			}
		}
		
		// update (thisTime - lastTime = timeElapsed)

		displayFrameCounter();

		lastTime = thisTime;

		cudaRenderer(dev_vfb);
		cudaMemcpy(vfb_linear, dev_vfb, sizeof(Color) * RES_X * RES_Y, cudaMemcpyDeviceToHost);
		convertDeviceToHostBuffer();
		displayVFB(vfb);
	}

	// free memory	
	freeDeviceMemory();
	cudaFree(dev_vfb);

	closeGraphics();

#else
	// capture the start time
	cudaEvent_t start, stop;
	cudaStartTimer(start, stop);

	// - InitializeScene
	initScene();

	// call kernels
	// - RenderScene
	cudaRenderer(dev_vfb);

	// memcpy DeviceToHost
	cudaMemcpy(vfb_linear, dev_vfb, sizeof(Color) * RES_X * RES_Y, cudaMemcpyDeviceToHost);

	// get stop time, and display the timing results
	cudaStopTimer(start, stop);

	// free memory	
	freeDeviceMemory();
	cudaFree(dev_vfb);

	convertDeviceToHostBuffer();	
	
	displayVFB(vfb);

	handleUserInput();
	
	closeGraphics();

#endif

	return EXIT_SUCCESS;
}