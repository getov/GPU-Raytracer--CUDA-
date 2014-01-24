#include "EventHandler.cuh"
#include <SDL\SDL.h>
#include "RaytracerControls.cuh"

EventHandler::EventHandler()
	: isRealTimeRendering(true)
{
	for (int i = 0; i < 323; ++i)
	{
		keysHeld[i] = false;
	}
}

void EventHandler::handleEvents()
{
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		handleKeyboard(event);
		handleMouse(event);
	}

	if (keysHeld[SDLK_w])
	{
		moveForward();
	}

	if (keysHeld[SDLK_s])
	{
		moveBackward();
	}

	if (keysHeld[SDLK_a])
	{
		strafeLeft();
	}

	if (keysHeld[SDLK_d])
	{
		strafeRight();
	}
}

void EventHandler::handleKeyboard(SDL_Event& ev)
{
	if (ev.type == SDL_KEYDOWN)
	{
		keysHeld[ev.key.keysym.sym] = true;
	}
	if (ev.type == SDL_KEYUP)
	{
		keysHeld[ev.key.keysym.sym] = false;
	}
	if (ev.type == SDL_QUIT)
	{
		isRealTimeRendering = false;
	}

	if (ev.key.keysym.sym == SDLK_ESCAPE)
	{
		isRealTimeRendering = false;
	}
}

void EventHandler::handleUserInput()
{
	SDL_Event event;
	while (SDL_WaitEvent(&event))
	{
		switch (event.type)
		{
			case SDL_QUIT:
				return;
			case SDL_KEYDOWN:
			{
				switch (event.key.keysym.sym)
				{
					case SDLK_ESCAPE:
						return;
					default:
						break;
				}
			}
			default:
				break;
		}
	}
}

void EventHandler::handleMouse(SDL_Event& ev)
{
	if(ev.type == SDL_MOUSEMOTION)
	{
		int x, y;
		Uint8 buttons = SDL_GetMouseState(&x, &y);

		if(buttons)
		{
			return;
		}

		float deltax = float(ev.motion.xrel);
		float deltay = float(ev.motion.yrel);

		setCameraOrientation(deltay, deltax);
	}
}