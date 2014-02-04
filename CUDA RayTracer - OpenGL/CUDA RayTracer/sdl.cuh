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
#ifndef SDL_H
#define SDL_H

#include "Color.cuh"
#include "Settings.cuh"

void initColorCache();
bool initGraphics(int frameWidth, int frameHeight);
void closeGraphics(void);
void displayVFB(Color vfb[VFB_MAX_SIZE][VFB_MAX_SIZE]); //!< displays the VFB (Virtual framebuffer) to the real one.
int frameWidth(void); //!< returns the frame width (pixels)
int frameHeight(void); //!< returns the frame height (pixels)

#endif // SDL_H
