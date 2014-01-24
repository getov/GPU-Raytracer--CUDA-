#include "EventHandler.cuh"
#include <SDL\SDL.h>
#include "RaytracerControls.cuh"
#include "Settings.cuh"

EventHandler::EventHandler()
	: isRealTimeRendering(true)
	, mouseSensitivity(0.6f)
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

	if (keysHeld[SDLK_KP_PLUS])
	{
		mouseSensitivity += 0.02;
	}
	else if (keysHeld[SDLK_KP_MINUS])
	{
		mouseSensitivity -= 0.02;

		if (mouseSensitivity <= 0)
		{
			mouseSensitivity = 0.6;
		}
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

	if (ev.type == SDL_KEYDOWN)
	{
		if (ev.key.keysym.sym == SDLK_F1)
		{
			GlobalSettings::AAEnabled = true;
		}
		else if (ev.key.keysym.sym == SDLK_F2)
		{
			GlobalSettings::AAEnabled = false;
		}

		// bugging
		//if (ev.key.keysym.sym == SDLK_F3 && GlobalSettings::AAEnabled)
		//{			
		//	GlobalSettings::previewAA = true;
		//}
		//else if (ev.key.keysym.sym == SDLK_F4)
		//{
		//	GlobalSettings::previewAA = false;
		//}
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

		float deltaX = float(ev.motion.xrel);
		float deltaY = float(ev.motion.yrel);

		SDL_ShowCursor(0);
		SDL_WM_GrabInput(SDL_GRAB_ON);

		setCameraOrientation(deltaY * mouseSensitivity, deltaX * mouseSensitivity);
	}
}