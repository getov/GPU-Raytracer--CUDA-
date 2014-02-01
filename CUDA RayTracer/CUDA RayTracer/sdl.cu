/***************************************************************************
 *   Copyright (C) 2009-2013 by Veselin Georgiev, Slavomir Kaslev et al    *
 *   admin@raytracing-bg.net                                               *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include <SDL.h>
#include <stdio.h>
#include "sdl.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

SDL_Surface* screen = NULL;

/// try to create a frame window with the given dimensions
bool initGraphics(int frameWidth, int frameHeight)
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("Cannot initialize SDL: %s\n", SDL_GetError());
		return false;
	}
	screen = SDL_SetVideoMode(frameWidth, frameHeight, 32, 0);

	if (!screen)
	{
		printf("Cannot set video mode %dx%d - %s\n", frameWidth, frameHeight, SDL_GetError());
		return false;
	}
	return true;
}

/// closes SDL graphics
void closeGraphics(void)
{
	SDL_Quit();
}

/// displays a VFB (virtual frame buffer) to the real framebuffer, with the necessary color clipping
void displayVFB(Color vfb[RES_Y][RES_X])
{
	int rs = screen->format->Rshift;
	int gs = screen->format->Gshift;
	int bs = screen->format->Bshift;

	for (int y = 0; y < screen->h; y++)
	{
		Uint32* row = (Uint32*) ((Uint8*) screen->pixels + y * screen->pitch);
		for (int x = 0; x < screen->w; x++)
			row[x] = vfb[y][x].toRGB32(rs, gs, bs);
	}
	SDL_Flip(screen);
}

/// returns the frame width
int frameWidth(void)
{
	if (screen) return screen->w;
	return 0;
}

/// returns the frame height
int frameHeight(void)
{
	if (screen) return screen->h;
	return 0;
}
