#ifndef EVENT_HANDLER_H
#define EVENT_HANDLER_H

#include <SDL.h>
#include <vector>

class EventHandler
{
private:
	EventHandler(const EventHandler& ev);
	EventHandler& operator=(const EventHandler& ev);

	// array that registers if a keyboard key is held
	std::vector<bool> keysHeld;

	// helper array for implementing 'on' and 'off' switch button system
	std::vector<bool> keySwitch;

	float mouseSensitivity;

public:
	EventHandler();

	bool isRealTimeRendering;

	/**
	* @brief - Combines handleKeyboard() and handleMouse()
	* The function registers keyboard presses
	* and mouse motion using SDL event system
	* @reference - handleKeyboard(SDL_Event& ev)
	* @reference - handleMouse(SDL_Event& ev)
	*/
	void handleEvents();

	/**
	* @brief - Gets the keyboard input from the user
	* @param ev - SDL_Event structure for handling the events
	*/
	void handleKeyboard(SDL_Event& ev);

	/**
	* @brief - Gets the mouse input from the user
	* @param ev - SDL_Event structure for handling the events
	*/
	void handleMouse(SDL_Event& ev);

	/**
	* @brief - Handles the keyboard input from the user
	* when the raytracer is not running in real-time mod
	*/
	void handleUserInput();
};

#endif