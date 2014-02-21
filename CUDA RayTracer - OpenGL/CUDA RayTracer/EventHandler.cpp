#include "EventHandler.h"
#include <SDL.h>
#include "RaytracerControls.cuh"
#include "Settings.cuh"
#include <Windows.h>

EventHandler::EventHandler()
	: isRealTimeRendering(true)
	, mouseSensitivity(0.6f)
	, scaleFactor(1.2)
	, angle(30.0)
	, translateFactor(5.0)
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

	// Object Transformations	
	if (GlobalSettings::isEditingAllowed)
	{
		// scaling
		if (glfwGetKey(GLFW_KEY_KP_1) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			scaleX(scaleFactor);
		}
		else if (glfwGetKey(GLFW_KEY_KP_1) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			scaleX(1.0 / scaleFactor);
		}

		if (glfwGetKey(GLFW_KEY_KP_2) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			scaleY(scaleFactor);
		}
		else if (glfwGetKey(GLFW_KEY_KP_2) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			scaleY(1.0 / scaleFactor);
		}

		if (glfwGetKey(GLFW_KEY_KP_3) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			scaleZ(scaleFactor);
		}
		else if (glfwGetKey(GLFW_KEY_KP_3) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			scaleZ(1.0 / scaleFactor);
		}

		// rotation
		if (glfwGetKey(GLFW_KEY_KP_4) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			rotateAroundX(angle);
		}
		else if (glfwGetKey(GLFW_KEY_KP_4) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			rotateAroundX(-angle);
		}

		if (glfwGetKey(GLFW_KEY_KP_5) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			rotateAroundY(angle);
		}
		else if (glfwGetKey(GLFW_KEY_KP_5) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			rotateAroundY(-angle);
		}

		if (glfwGetKey(GLFW_KEY_KP_6) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			rotateAroundZ(angle);
		}
		else if (glfwGetKey(GLFW_KEY_KP_6) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			rotateAroundZ(-angle);
		}

		// translation
		if (glfwGetKey(GLFW_KEY_KP_7) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			translateX(translateFactor);
		}
		else if (glfwGetKey(GLFW_KEY_KP_7) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			translateX(-translateFactor);
		}

		if (glfwGetKey(GLFW_KEY_KP_8) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			translateY(translateFactor);
		}
		else if (glfwGetKey(GLFW_KEY_KP_8) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			translateY(-translateFactor);
		}

		if (glfwGetKey(GLFW_KEY_KP_9) == GLFW_PRESS && 
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			translateZ(translateFactor);
		}
		else if (glfwGetKey(GLFW_KEY_KP_9) == GLFW_PRESS && 
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			translateZ(-translateFactor);
		}

		// set transformation precision
		if (glfwGetKey('1') == GLFW_PRESS)
		{
			scaleFactor		= 1.01;
			angle			= 2.0;
			translateFactor = 1.0; 
		}
		if (glfwGetKey('2') == GLFW_PRESS)
		{
			scaleFactor		= 1.015;
			angle			= 4.5;
			translateFactor = 1.25; 
		}
		if (glfwGetKey('3') == GLFW_PRESS)
		{
			scaleFactor		= 1.02;
			angle			= 8.0;
			translateFactor = 1.6; 
		}
		if (glfwGetKey('4') == GLFW_PRESS)
		{
			scaleFactor		= 1.10;
			angle			= 15.0;
			translateFactor = 2.5; 
		}
		if (glfwGetKey('5') == GLFW_PRESS)
		{
			scaleFactor		= 1.2;
			angle			= 30.0;
			translateFactor = 5.0; 
		}

		/*if (glfwGetKey('1') == GLFW_PRESS &&
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			scaleFactor += 0.01;
		}
		else if (glfwGetKey('1') == GLFW_PRESS &&
				 glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			scaleFactor -= 0.01;
			if (scaleFactor < 1.0)
			{
				scaleFactor = 1.0;
			}
		}*/
		
		// Ligth Movement
		if (glfwGetKey('J') == GLFW_PRESS &&
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			// +X direction
			moveLightX(1);
		}
		else if (glfwGetKey('J') == GLFW_PRESS &&
			     glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			// -X direction
			moveLightX(-1);
		}

		if (glfwGetKey('K') == GLFW_PRESS &&
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			// +Y direction
			moveLightY(1);
		}
		else if (glfwGetKey('K') == GLFW_PRESS &&
			     glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			// -Y direction
			moveLightY(-1);
		}

		if (glfwGetKey('L') == GLFW_PRESS &&
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			// +Z direction
			moveLightZ(1);
		}
		else if (glfwGetKey('L') == GLFW_PRESS &&
			     glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			// -Z direction
			moveLightZ(-1);
		}

		if (glfwGetKey('P') == GLFW_PRESS &&
			glfwGetKey(GLFW_KEY_LALT) == GLFW_RELEASE)
		{
			// increase light power
			regulateLightPower(1);
		}
		else if (glfwGetKey('P') == GLFW_PRESS &&
			     glfwGetKey(GLFW_KEY_LALT) == GLFW_PRESS)
		{
			// decrease light power
			regulateLightPower(-1);
		}
	}

	// EXIT
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

bool EventHandler::keySwitch[4] = { false, false, false, false };

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

		if (key == GLFW_KEY_KP_0 && !keySwitch[3])
		{
			GlobalSettings::isEditingAllowed = true;
			keySwitch[3] = true;
		}
		else if (key == GLFW_KEY_KP_0 && keySwitch[3])
		{
			GlobalSettings::isEditingAllowed = false;
			keySwitch[3] = false;
		}

		if (GlobalSettings::isEditingAllowed)
		{
			// target geometries
			if (glfwGetKey(GLFW_KEY_UP) == GLFW_PRESS)
			{
				targetNextGeometry();
			}
			else if (glfwGetKey(GLFW_KEY_DOWN) == GLFW_PRESS)
			{
				targetPreviousGeometry();		
			}
		}
		else
		{
			discardSelectedNode();
		}

		// open controls.txt
		if (glfwGetKey(GLFW_KEY_F12) == GLFW_PRESS)
		{
			ShellExecute(0, 0, "controls.txt", 0, 0, SW_SHOW);
		}
	}	
}