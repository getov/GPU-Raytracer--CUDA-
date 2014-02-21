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
extern "C" void updateScene(const double& elapsedTime, const double& currentTime);

unsigned frameCount;
unsigned lastFrameEnd;
unsigned lastTitleUpdateTime;
unsigned lastTitleUpdateFrameCount;
const char* const appName = "CUDA Traycer";

//* Data handles to the OpenGL buffer
GLuint bufferObj;
cudaGraphicsResource* resource;

/**
 * @brief - render the scene using glDrawPixels()
 * and then swap the buffers using glfwSwapBuffers()
*/
static void glRenderScene()
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
	glfwSetWindowTitle(info);

	cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * @brief - Calculate and display on the window frame
 * the mean frame time and the frames per second.
 * Using glfwGetTime() to register the time (in seconds).
*/
void displayFrameCounter()
{
	++frameCount;

	const unsigned now = glfwGetTime();
	const unsigned frameTime = now - lastFrameEnd;
	const unsigned titleUpdateTimeDelta = now - lastTitleUpdateTime;

	if (titleUpdateTimeDelta > 1)
	{
		const unsigned framesDelta = frameCount - lastTitleUpdateFrameCount;
		const unsigned meanFrameTime = titleUpdateTimeDelta / framesDelta;
		const unsigned fps = framesDelta  / titleUpdateTimeDelta;

		std::ostringstream title;
		title << appName << " :\t\t\t mean frame time: " << meanFrameTime << " ms || fps: " << fps;
		title.flush();

		glfwSetWindowTitle(title.str().c_str());

		lastTitleUpdateTime = now;
		lastTitleUpdateFrameCount = frameCount;
	}

	lastFrameEnd = glfwGetTime();
}

/**
 * Set CUDA device (GPU)
 * Initialize GLFW and GLEW and open GLFW window
 * Generate and bind the buffer data
*/
void OpenGL_Setup();

/**
 * @brief - Perform a clenup that:
 * - Unregisters the CUDA recource
 * - Unbinds the OpenGL buffer
 * - Deletes the OpenGL buffer
*/
void Clean_OpenGL_and_CUDA();

int main(int argc, char** argv)
{
	cudaDeviceSetLimit(cudaLimitStackSize, STACK_SIZE);

	Menu mainMenu(appName);
	mainMenu.Destroy();

	printGPUSpecs();

	EventHandler eventController;

	OpenGL_Setup();

	//* framebuffer used by the GPU
	uchar4* dev_vfb;
	size_t size;

	initScene();

	if (GlobalSettings::realTime)
	{
		double lastTime = glfwGetTime();

		while (glfwGetWindowParam(GLFW_OPENED))
		{
			double thisTime = glfwGetTime();

			updateScene(thisTime - lastTime, thisTime);
			
			// notify CUDA runtime that we want to share bufferObj with resource (CUDA)
			cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone );

			cudaGraphicsMapResources( 1, &resource, NULL );
			
			// map the addres of 'resource' to 'dev_vfb'
			cudaGraphicsResourceGetMappedPointer( (void**)&dev_vfb, 
												  &size, 
												  resource);

			cameraBeginFrame();

			displayFrameCounter();

			cudaRenderer(dev_vfb);

			// unmap resource so the CUDA and OpenGL buffers can synchronisze
			cudaGraphicsUnmapResources( 1, &resource, NULL );

			eventController.handleEvents();
			glfwSetKeyCallback(eventController.keyboardCallback);

			lastTime = thisTime;

			glRenderScene();
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

		cudaEvent_t start, stop;

		cudaStartTimer(start, stop);

		cudaRenderer(dev_vfb);

		cudaGraphicsUnmapResources( 1, &resource, NULL );

		glRenderScene();

		cudaStopTimer(start, stop);

			
		while (glfwGetWindowParam(GLFW_OPENED))
		{
			eventController.handleUserInput();

			glfwWaitEvents();
		}
		
	}

	freeDeviceMemory();

	Clean_OpenGL_and_CUDA();

	glfwTerminate();

	return EXIT_SUCCESS;
}

void OpenGL_Setup()
{
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
	
    if(!glfwOpenWindow(GlobalSettings::RES_X, GlobalSettings::RES_Y, 8, 8, 8, 8, 16, 0,
		               GlobalSettings::fullscreen ? GLFW_FULLSCREEN : GLFW_WINDOW))
	{
		std::cerr << "glfwOpenWindow failed!\n";
	}

	glewExperimental = GL_TRUE;
	
	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Failed to initialize GLEW\n";
	}

	while(glGetError() != GL_NO_ERROR) {}

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB,
				  GlobalSettings::RES_X * GlobalSettings::RES_Y * 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );
}

void Clean_OpenGL_and_CUDA()
{
	cudaGraphicsUnregisterResource( resource );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	glDeleteBuffers( 1, &bufferObj );
}