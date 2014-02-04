#ifndef SETTINGS_H
#define SETTINGS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define VFB_MAX_SIZE 1920
#define INF 1e99
#define MAX_RAY_DEPTH 5
#define PI 3.141592653589793238

enum SceneID { CORNELL_BOX, ROAMING, SEA };

struct GlobalSettings
{
	static bool AAEnabled;
	static bool previewAA;
	static bool realTime;
	static bool fullscreen;
	static int RES_X;
	static int RES_Y;
	static short sceneID; 
};

//__device__
//const Color ambientLight(0.2, 0.2, 0.2);

// the stack size in bytes for each GPU thread
const size_t STACK_SIZE = 20 * 1024;

const int GEOM_MAX_SIZE = 1024;

__device__ 
static int GEOM_COUNT = 0;

#endif // SETTINGS_H