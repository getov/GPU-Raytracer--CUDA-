#ifndef EVENT_HANDLER_H
#define EVENT_HANDLER_H

#include <SDL/SDL.h>

class EventHandler
{
private:
	EventHandler(const EventHandler& ev);
	EventHandler& operator=(const EventHandler& ev);

	bool keysHeld[323];

	float mouseSensitivity;

public:
	EventHandler();

	bool isRealTimeRendering;

	void handleEvents();
	void handleKeyboard(SDL_Event& ev);
	void handleMouse(SDL_Event& ev);

	// used when not real time rendering
	void handleUserInput();
};

#endif