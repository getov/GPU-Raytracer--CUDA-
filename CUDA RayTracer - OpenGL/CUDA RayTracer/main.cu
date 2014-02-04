#include <GL\glew.h>
#include <GL\glfw.h>
#include <GL\freeglut.h>
#include <GL\GL.h>

#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_gl_interop.h"




#include <SDL.h>
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
#include "EventHandler.h"
#include "Menu.h"
#include "Settings.cuh"
#include "RaytracerControls.cuh"
	
using namespace std;

extern "C" void cudaRenderer(uchar4* dev_vfb);
extern "C" void freeDeviceMemory();
extern "C" void initScene();
extern "C" void cameraBeginFrame();


unsigned frameCount;
unsigned lastFrameEnd;
unsigned lastTitleUpdateTime;
unsigned lastTitleUpdateFrameCount;
const char* const appName = "CUDA Traycer";

GLuint bufferObj;
cudaGraphicsResource* resource;

static void draw(void)
{
	glDrawPixels(GlobalSettings::RES_X, GlobalSettings::RES_Y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glfwSwapBuffers();
}


/**
 * @brief - Function that prints CUDA specs 
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
 * @brief - Wrapper function that creates timer and captures the start and stop time
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
 * @brief - Wrapper function that takes the previously captured start and stop time
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
	sprintf(info, "CUDA Traycer || Time to render: %3.1f ms", elapsedTime);
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
		title << appName << " :\t\t\t mean frame time: " << meanFrameTime << " ms || fps: " << fps;
		title.flush();

		glfwSetWindowTitle(title.str().c_str());

		lastTitleUpdateTime = now;
		lastTitleUpdateFrameCount = frameCount;
	}

	lastFrameEnd = SDL_GetTicks();
}

int main(int argc, char** argv)
{
	Menu mainMenu(appName);
	mainMenu.Destroy();

	EventHandler eventController;

	cudaDeviceProp prop;
	int device;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&device, &prop);
	cudaGLSetGLDevice(device);

	if (!glfwInit())
	{
		std::cerr << "Could not initialize GLFW\n";
	}

	// open a window with GLFW
    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE); //GLFW_OPENGL_CORE_PROFILE , GLFW_OPENGL_COMPAT_PROFILE
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwOpenWindowHint(GLFW_WINDOW_NO_RESIZE, GL_TRUE);
    if(!glfwOpenWindow(GlobalSettings::RES_X, GlobalSettings::RES_Y, 8, 8, 8, 8, 16, 0, GlobalSettings::fullscreen ? GLFW_FULLSCREEN : GLFW_WINDOW)) // for full screen = GLFW_FULLSCREEN
	{
		std::cerr << "glfwOpenWindow failed!\n";
	}

	glewExperimental = GL_TRUE;
	// initialize GLEW
	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Failed to initialize GLEW\n";
	}

	while(glGetError() != GL_NO_ERROR) {}

	/*glfwDisable(GLFW_MOUSE_CURSOR);
    glfwSetMousePos(0, 0);
    glfwSetMouseWheel(0);*/

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, GlobalSettings::RES_X * GlobalSettings::RES_Y * 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );

	/*cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone );

	cudaGraphicsMapResources( 1, &resource, NULL );*/
	uchar4* dev_vfb;
	size_t size;
	/*cudaGraphicsResourceGetMappedPointer( (void**)&dev_vfb, 
                                              &size, 
                                              resource);*/

	cudaDeviceSetLimit(cudaLimitStackSize, STACK_SIZE);

	// InitializeScene
	initScene();


	if (GlobalSettings::realTime)
	{
		while (glfwGetWindowParam(GLFW_OPENED))
		{
			cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone );

			cudaGraphicsMapResources( 1, &resource, NULL );
			
			cudaGraphicsResourceGetMappedPointer( (void**)&dev_vfb, 
												  &size, 
												  resource);

			cameraBeginFrame();

			displayFrameCounter();

			

			
			cudaRenderer(dev_vfb);

			cudaGraphicsUnmapResources( 1, &resource, NULL );

			//eventController.handleEvents();
			if (glfwGetKey('W'))
			{
				moveForward();
				printf("w");
			}

			if (glfwGetKey('S'))
			{
				moveBackward();
			}

			if (glfwGetKey('A'))
			{
				strafeLeft();
			}

			if (glfwGetKey('D'))
			{
				strafeRight();
			}

			if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS)
			{
				glfwTerminate();
			}

			int mouseX, mouseY;
			glfwGetMousePos(&mouseX, &mouseY);
			setCameraOrientation(mouseY * 0.6, mouseX * 0.6);

			glfwDisable(GLFW_MOUSE_CURSOR);
			glfwSetMousePos(0, 0);
			
			draw();
		}
	}
	else
	{
		cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone );

		cudaGraphicsMapResources( 1, &resource, NULL );
			
		cudaGraphicsResourceGetMappedPointer( (void**)&dev_vfb, 
												&size, 
												resource);
		while (glfwGetWindowParam(GLFW_OPENED))
		{
			cudaEvent_t start, stop;
			cudaStartTimer(start, stop);
			displayFrameCounter();
			cudaGraphicsResourceGetMappedPointer( (void**)&dev_vfb, 
												  &size, 
												  resource);

			cudaRenderer(dev_vfb);

			cudaGraphicsUnmapResources( 1, &resource, NULL );

			draw();
			cudaStopTimer(start, stop);
		}
	}

	
	// free memory	
	freeDeviceMemory();
	glfwTerminate();

	return EXIT_SUCCESS;
}