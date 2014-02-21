#ifndef SETTINGS_H
#define SETTINGS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define VFB_MAX_SIZE 1920
#define INF 1e99
#define MAX_RAY_DEPTH 5
#define PI 3.141592653589793238

//* enum with scene IDs
enum SceneID { CORNELL_BOX, ROAMING, SEA };

/**
 * @struct GlobalSettings
 * @brief - Structure that stores all the 
 * global settings for the raytracer
*/
struct GlobalSettings
{
	__device__ GlobalSettings(){};
	static bool AAEnabled;
	static bool previewAA;
	static bool blur;
	static bool grayscale;
	static bool realTime;
	static bool fullscreen;
	static bool isEditingAllowed;
	static int RES_X;
	static int RES_Y;
	static short sceneID; 
};

// the stack size in bytes for each GPU thread
const size_t STACK_SIZE = 20 * 1024;

#endif // SETTINGS_H