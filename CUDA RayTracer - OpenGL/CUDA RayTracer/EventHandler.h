#ifndef EVENT_HANDLER_H
#define EVENT_HANDLER_H

#include <GL\glfw.h>
#include <vector>

class EventHandler
{
private:
	EventHandler(const EventHandler& ev);
	EventHandler& operator=(const EventHandler& ev);

	// helper array for implementing 'on' and 'off' switch button system
	std::vector<bool> keySwitch;

	float mouseSensitivity;

public:
	EventHandler();

	bool isRealTimeRendering;

	/**
	* @brief - Combines handleKeyboard() and handleMouse()
	* The function registers keyboard presses
	* and mouse motion using GLFW
	* @reference - handleKeyboard()
	* @reference - handleMouse()
	*/
	void handleEvents();

	/**
	* @brief - Gets the keyboard input from the user
	*/
	void handleKeyboard();

	/**
	* @brief - Gets the mouse input from the user
	*/
	void handleMouse();

	/**
	* @brief - Handles the keyboard input from the user
	* when the raytracer is not running in real-time mod
	*/
	void handleUserInput();
};

#endif