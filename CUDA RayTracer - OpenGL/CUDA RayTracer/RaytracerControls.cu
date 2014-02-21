#include "RaytracerControls.cuh"
#include "CameraController.cuh"
#include "cuda_renderer.cuh"
#include "Sphere.cuh"
#include "Lambert.cuh"

extern __device__
CameraController* controller;

// movement
__global__ void mvForward()
{
	controller->moveForward();
}
extern "C" void moveForward()
{
	mvForward<<<1, 1>>>();
}

__global__ void mvBackward()
{
	controller->moveBackward();
}
extern "C" void moveBackward()
{
	mvBackward<<<1, 1>>>();
}

__global__ void mvLeft()
{
	controller->strafeLeft();
}
extern "C" void strafeLeft()
{
	mvLeft<<<1, 1>>>();
}

__global__ void mvRight()
{
	controller->strafeRight();
}
extern "C" void strafeRight()
{
	mvRight<<<1, 1>>>();
}

// rotation
__global__ 
void setCamOrientation(float zenith, float azimuth)
{
	controller->offsetCameraOrientation(zenith, azimuth);
}

extern "C"
void setCameraOrientation(float zenith, float azimuth)
{
	setCamOrientation<<<1, 1>>>(zenith, azimuth);
}

// Target Geometries
__global__ void target_next_geom()
{
	++(scene->indexGeom);

	if (scene->indexGeom >= scene->dev_nodes.size())
	{
		scene->indexGeom = 0;
	}

	scene->selectedNode = scene->dev_nodes[scene->indexGeom];
}
extern "C" void targetNextGeometry()
{
	target_next_geom<<<1, 1>>>();
}

__global__ void target_prev_geom()
{
	--(scene->indexGeom);

	if (scene->indexGeom < 0)
	{
		scene->indexGeom = scene->dev_nodes.size() - 1;
	}

	scene->selectedNode = scene->dev_nodes[scene->indexGeom];
}
extern "C" void targetPreviousGeometry()
{
	target_prev_geom<<<1, 1>>>();
}

// Object transformations
__global__ void scale_x(double scaleFactor)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.scale(scaleFactor, 1, 1);
	}
}
extern "C" void scaleX(double scaleFactor)
{
	scale_x<<<1, 1>>>(scaleFactor);
}

__global__ void scale_z(double scaleFactor)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.scale(1, 1, scaleFactor);
	}
}
extern "C" void scaleZ(double scaleFactor)
{
	scale_z<<<1, 1>>>(scaleFactor);
}

__global__ void scale_y(double scaleFactor)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.scale(1, scaleFactor, 1);
	}
}
extern "C" void scaleY(double scaleFactor)
{
	scale_y<<<1, 1>>>(scaleFactor);
}

__global__ void rotate_x(double angle)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.rotate(0, angle, 0);
	}
}
extern "C" void rotateAroundX(double angle)
{
	rotate_x<<<1, 1>>>(angle);
}

__global__ void rotate_y(double angle)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.rotate(angle, 0, 0);
	}
}
extern "C" void rotateAroundY(double angle)
{
	rotate_y<<<1, 1>>>(angle);
}

__global__ void rotate_z(double angle)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.rotate(0, 0, angle);
	}
}
extern "C" void rotateAroundZ(double angle)
{
	rotate_z<<<1, 1>>>(angle);
}

__global__ void translate_x(double translateFactor)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.translate(
			scene->selectedNode->transform.point(Vector(translateFactor, 0, 0)));
	}
}
extern "C" void translateX(double translateFactor)
{
	translate_x<<<1, 1>>>(translateFactor);
}

__global__ void translate_y(double translateFactor)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.translate(
			scene->selectedNode->transform.point(Vector(0, translateFactor, 0)));
	}
}
extern "C" void translateY(double translateFactor)
{
	translate_y<<<1, 1>>>(translateFactor);
}

__global__ void translate_z(double translateFactor)
{
	if (scene->selectedNode)
	{
		scene->selectedNode->transform.translate(
			scene->selectedNode->transform.point(Vector(0, 0, translateFactor)));
	}
}
extern "C" void translateZ(double translateFactor)
{
	translate_z<<<1, 1>>>(translateFactor);
}

__global__ void null_current_node()
{
	scene->selectedNode = nullptr;
}
extern "C" void discardSelectedNode()
{
	null_current_node<<<1, 1>>>();
}

// Light
__global__ void move_light_x(int dir)
{
	if (!scene->dev_lights.empty())
	{
		scene->dev_lights[0]->setPosition(
				scene->dev_lights[0]->getPosition() + Vector(dir, 0, 0) * 1.5);
	}	
}
extern "C" void moveLightX(int dir)
{
	move_light_x<<<1, 1>>>(dir);
}

__global__ void move_light_z(int dir)
{
	if (!scene->dev_lights.empty())
	{
		scene->dev_lights[0]->setPosition(
				scene->dev_lights[0]->getPosition() + Vector(0, 0, dir) * 1.5);
	}
}
extern "C" void moveLightZ(int dir)
{
	move_light_z<<<1, 1>>>(dir);
}

__global__ void move_light_y(int dir)
{
	if (!scene->dev_lights.empty())
	{
		scene->dev_lights[0]->setPosition(
				scene->dev_lights[0]->getPosition() + Vector(0, dir, 0) * 1.5);
	}
}
extern "C" void moveLightY(int dir)
{
	move_light_y<<<1, 1>>>(dir);
}

__global__ void reg_light_power(int mult)
{
	if (!scene->dev_lights.empty())
	{
		scene->dev_lights[0]->regulatePower(mult * 1000.0);
	}
}
extern "C" void regulateLightPower(int mult)
{
	reg_light_power<<<1, 1>>>(mult);
}

// Shader selection
__global__ void get_next_shader()
{
	if (scene->selectedNode)
	{
		++(scene->indexShader);

		if (scene->indexShader >= scene->dev_shaders.size())
		{
			scene->indexShader = 0;
		}

		scene->selectedNode->shader = scene->dev_shaders[scene->indexShader];
	}
}
extern "C" void getNextShader()
{
	get_next_shader<<<1, 1>>>();
}

__global__ void get_prev_shader()
{
	if (scene->selectedNode)
	{
		--(scene->indexShader);

		if (scene->indexShader < 0)
		{
			scene->indexShader = scene->dev_shaders.size() - 1;
		}

		scene->selectedNode->shader = scene->dev_shaders[scene->indexShader];
	}
}
extern "C" void getPreviousShader()
{
	get_prev_shader<<<1, 1>>>();
}

// generate geometry
__global__ void gen_sphere()
{
	Node* newNode = createNode(new Sphere(Vector(0, 0, 0), 40.0),
							   new Lambert(Color(0.9, 0.9, 0.9)));
	newNode->transform.translate(scene->dev_cam->pos + scene->dev_cam->frontDir * 120.0);
}
extern "C" void generateSphere()
{
	gen_sphere<<<1, 1>>>();
}