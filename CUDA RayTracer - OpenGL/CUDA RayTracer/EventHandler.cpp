#include "EventHandler.h"
#include <SDL.h>
#include "RaytracerControls.cuh"
#include "Settings.cuh"

EventHandler::EventHandler()
	: isRealTimeRendering(true)
	, mouseSensitivity(0.6f)
{
}

void EventHandler::handleEvents()
{
	handleKeyboard();
	handleMouse();
}

void EventHandler::handleKeyboard()
{
	if (glfwGetKey('W'))
	{
		moveForward();
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

	if (glfwGetKey(GLFW_KEY_KP_ADD) == GLFW_PRESS)
	{
		mouseSensitivity += 0.02;
	}
	else if (glfwGetKey(GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
	{
		mouseSensitivity -= 0.02;

		if (mouseSensitivity <= 0)
		{
			mouseSensitivity = 0.6;
		}
	}

	if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS)
	{
		glfwTerminate();
	}
}

void EventHandler::handleUserInput()
{
	if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS)
	{
		glfwTerminate();
	}
}

void EventHandler::handleMouse()
{
	int mouseX, mouseY;

	glfwGetMousePos(&mouseX, &mouseY);

	setCameraOrientation(mouseY * mouseSensitivity, mouseX * mouseSensitivity);

	glfwDisable(GLFW_MOUSE_CURSOR);
	glfwSetMousePos(0, 0);
}

bool EventHandler::keySwitch[3] = { false, false, false };

void GLFWCALL EventHandler::keyboardCallback(int key, int action)
{	
	if( action == GLFW_PRESS )
	{
		if (key == GLFW_KEY_F1 && !keySwitch[0])
		{
			GlobalSettings::AAEnabled = !GlobalSettings::AAEnabled;
			keySwitch[0] = true;
		}
		else if (key == GLFW_KEY_F1 && keySwitch[0])
		{
			GlobalSettings::AAEnabled = !GlobalSettings::AAEnabled;
			keySwitch[0] = false;
		}

		if (key == GLFW_KEY_F2 && !keySwitch[1])
		{
			GlobalSettings::blur = !GlobalSettings::blur;
			keySwitch[1] = true;
		}
		else if (key == GLFW_KEY_F2 && keySwitch[1])
		{
			GlobalSettings::blur = !GlobalSettings::blur;
			keySwitch[1] = false;
		}

		if (key == GLFW_KEY_F3 && !keySwitch[2])
		{
			GlobalSettings::grayscale = !GlobalSettings::grayscale;
			keySwitch[2] = true;
		}
		else if (key == GLFW_KEY_F3 && keySwitch[2])
		{
			GlobalSettings::grayscale = !GlobalSettings::grayscale;
			keySwitch[2] = false;
		}
	}	
}