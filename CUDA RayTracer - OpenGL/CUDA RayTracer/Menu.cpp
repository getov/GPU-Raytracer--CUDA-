#include "Menu.h"
#include "Settings.cuh"
#include <Windows.h>

Button::Button(const char* buttonName, const SDL_Color& buttonColor, const int& buttonID, int fontSize)
	: name(buttonName)
	, color(buttonColor)
	, id(buttonID)
	, _fontSize(fontSize)
{
	font = TTF_OpenFont("../Fonts/Starcraft Normal.ttf", _fontSize);

	body = TTF_RenderText_Solid(font, name, color);
}

Menu::Menu(const char* name)
	: selectedItem(2, false)
	, hoveredItem(2, false)
	, appName(name)
{
	SDL_Init(SDL_INIT_VIDEO);
	
	menuScreen = SDL_SetVideoMode(1280, 720, 32, 0);

	SDL_WM_SetCaption(appName, NULL);

	TTF_Init();

	setTextColor();

	renderMenuLabels();
	renderOptionButtons();

	selectedItem.resize(optionButtons.size(), false);
	hoveredItem.resize(optionButtons.size(), false);

	SDL_FillRect(menuScreen, &menuScreen->clip_rect, SDL_MapRGB(menuScreen->format, 0x00, 0x00, 0x00));

	handleMouseEvents();
}

void Menu::setTextColor()
{
	// original color
	color[0].r = 95;
	color[0].g = 95;
	color[0].b = 95;

	// hover color
	color[1].r = 195;
	color[1].g = 195;
	color[1].b = 195;

	// selected color
	color[2].r = 0;
	color[2].g = 255;
	color[2].b = 0;

	// white
	color[3].r = 255;
	color[3].g = 255;
	color[3].b = 255;
}

void Menu::renderMenuLabels()
{
	menuLabels.push_back(new Button("Resolution", color[3], 0));
	menuLabels.push_back(new Button("Video Mode", color[3], 0));
	menuLabels.push_back(new Button("Anti-Aliasing", color[3], 0));
	menuLabels.push_back(new Button("Real-Time Rendering", color[3], 0));
	menuLabels.push_back(new Button("Choose Scene", color[3], 0));

	float offsetY = 0.0;
	for (int i = 0; i < menuLabels.size(); ++i)
	{
		menuLabels[i]->boundingRect.x = menuScreen->clip_rect.w * 0.025;
		menuLabels[i]->boundingRect.y = menuScreen->clip_rect.h * (0.3 + offsetY);
		offsetY += 0.1;
	}
}

void Menu::renderOptionButtons()
{
	// Resolution
	optionButtons.push_back(new Button("640x480", color[0], 1));
	optionButtons.push_back(new Button("1024x768", color[0], 2));
	optionButtons.push_back(new Button("1280x960", color[0], 3));
	optionButtons.push_back(new Button("1920x1080", color[0], 4));
	int lastSize = optionButtons.size();

	float offsetX = 0.0;
	for (int i = 0; i < lastSize; ++i)
	{
		optionButtons[i]->boundingRect.x = menuScreen->clip_rect.w * (0.35 + offsetX);
		optionButtons[i]->boundingRect.y = menuScreen->clip_rect.h * 0.3;
		offsetX += 0.15;
	}

	// Anti-Aliasing
	optionButtons.push_back(new Button("On", color[0], 5));
	optionButtons.push_back(new Button("Off", color[0], 6));

	offsetX = 0.0;
	for (int i = lastSize; i < optionButtons.size(); ++i)
	{
		optionButtons[i]->boundingRect.x = menuScreen->clip_rect.w * (0.35 + offsetX);
		optionButtons[i]->boundingRect.y = menuScreen->clip_rect.h * 0.5;
		offsetX += 0.05;
	}
	lastSize = optionButtons.size();

	// Real-Time Rendering
	optionButtons.push_back(new Button("On", color[0], 7));
	optionButtons.push_back(new Button("Off", color[0], 8));

	offsetX = 0.0;
	for (int i = lastSize; i < optionButtons.size(); ++i)
	{
		optionButtons[i]->boundingRect.x = menuScreen->clip_rect.w * (0.35 + offsetX);
		optionButtons[i]->boundingRect.y = menuScreen->clip_rect.h * 0.6;
		offsetX += 0.05;
	}

	// Render button
	optionButtons.push_back(new Button("Render", color[0], 9, 25));
	optionButtons[8]->boundingRect.x = menuScreen->clip_rect.w * 0.80;
	optionButtons[8]->boundingRect.y = menuScreen->clip_rect.h * 0.85;
	lastSize = optionButtons.size();

	// Video Mode
	optionButtons.push_back(new Button("Windowed", color[0], 10));
	optionButtons.push_back(new Button("Fullscreen", color[0], 11));
	
	offsetX = 0.0;
	for (int i = lastSize; i < optionButtons.size(); ++i)
	{
		optionButtons[i]->boundingRect.x = menuScreen->clip_rect.w * (0.35 + offsetX);
		optionButtons[i]->boundingRect.y = menuScreen->clip_rect.h * 0.4;
		offsetX += 0.15;
	}
	lastSize = optionButtons.size();

	// Scenes
	optionButtons.push_back(new Button("Cornell Box", color[0], 12));
	optionButtons.push_back(new Button("Roaming", color[0], 13));
	optionButtons.push_back(new Button("Island", color[0], 14));
	optionButtons.push_back(new Button("Roaming_v2", color[0], 15));

	offsetX = 0.0;
	for (int i = lastSize; i < optionButtons.size(); ++i)
	{
		optionButtons[i]->boundingRect.x = menuScreen->clip_rect.w * (0.35 + offsetX);
		optionButtons[i]->boundingRect.y = menuScreen->clip_rect.h * 0.7;
		offsetX += 0.15;
	}
	lastSize = optionButtons.size();

	// show controls
	optionButtons.push_back(new Button("Show Controls", color[0], 1337));
	optionButtons[lastSize]->boundingRect.x = menuScreen->clip_rect.w * 0.025;
	optionButtons[lastSize]->boundingRect.y = menuScreen->clip_rect.h * 0.9;
}

void Menu::handleMouseEvents()
{
	int x, y;
	SDL_Event ev;
	bool inOptionsMenu = true;
	while(inOptionsMenu)
	{
		while(SDL_PollEvent(&ev))
		{
			switch (ev.type)
			{
				case SDL_MOUSEBUTTONDOWN:
				{
					x = ev.button.x;
					y = ev.button.y;

					for (int i = 0; i < optionButtons.size(); ++i)
					{
						if ( x >= optionButtons[i]->boundingRect.x &&
							(x <= optionButtons[i]->boundingRect.x + optionButtons[i]->boundingRect.w) &&
							 y >= optionButtons[i]->boundingRect.y &&
							(y <= optionButtons[i]->boundingRect.y + optionButtons[i]->boundingRect.h) )
						{
							// Resolution
							if (optionButtons[i]->id == 1 && !selectedItem[i])
							{
								GlobalSettings::RES_X = 640;
								GlobalSettings::RES_Y = 480;

								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
																
								selectedItem[i + 1] = false;
								selectedItem[i + 2] = false;
								selectedItem[i + 3] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								SDL_FreeSurface(optionButtons[i + 2]->body);
								SDL_FreeSurface(optionButtons[i + 3]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
								optionButtons[i + 2]->body = TTF_RenderText_Solid(optionButtons[i + 2]->font, optionButtons[i + 2]->name, color[0]);
								optionButtons[i + 3]->body = TTF_RenderText_Solid(optionButtons[i + 3]->font, optionButtons[i + 3]->name, color[0]);

							}
							else if (optionButtons[i]->id == 2 && !selectedItem[i])
							{
								GlobalSettings::RES_X = 1024;
								GlobalSettings::RES_Y = 768;

								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								
								selectedItem[i + 1] = false;
								selectedItem[i + 2] = false;
								selectedItem[i - 1] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								SDL_FreeSurface(optionButtons[i + 2]->body);
								SDL_FreeSurface(optionButtons[i - 1]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
								optionButtons[i + 2]->body = TTF_RenderText_Solid(optionButtons[i + 2]->font, optionButtons[i + 2]->name, color[0]);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
							}
							else if (optionButtons[i]->id == 3 && !selectedItem[i])
							{
								GlobalSettings::RES_X = 1280;
								GlobalSettings::RES_Y = 960;

								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								
								selectedItem[i + 1] = false;
								selectedItem[i - 1] = false;
								selectedItem[i - 2] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								SDL_FreeSurface(optionButtons[i - 1]->body);
								SDL_FreeSurface(optionButtons[i - 2]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
								optionButtons[i - 2]->body = TTF_RenderText_Solid(optionButtons[i - 2]->font, optionButtons[i - 2]->name, color[0]);
							}
							else if (optionButtons[i]->id == 4 && !selectedItem[i])
							{
								GlobalSettings::RES_X = 1920;
								GlobalSettings::RES_Y = 1080;

								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);

								selectedItem[i - 1] = false;
								selectedItem[i - 2] = false;
								selectedItem[i - 3] = false;
								SDL_FreeSurface(optionButtons[i - 1]->body);
								SDL_FreeSurface(optionButtons[i - 2]->body);
								SDL_FreeSurface(optionButtons[i - 3]->body);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
								optionButtons[i - 2]->body = TTF_RenderText_Solid(optionButtons[i - 2]->font, optionButtons[i - 2]->name, color[0]);
								optionButtons[i - 3]->body = TTF_RenderText_Solid(optionButtons[i - 3]->font, optionButtons[i - 3]->name, color[0]);
							}

							// Anti-Aliasing - by default GlobalSettings::AAEnabled is true
							if (optionButtons[i]->id == 5 && !selectedItem[i]) // button -> ON
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::AAEnabled = true;

								selectedItem[i + 1] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
							}
							else if (optionButtons[i]->id == 6 && !selectedItem[i]) // button -> OFF
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::AAEnabled = false;

								selectedItem[i - 1] = false;
								SDL_FreeSurface(optionButtons[i - 1]->body);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
							}

							// Real-Time Rendering - by default Real-Time rendering is false
							if (optionButtons[i]->id == 7 && !selectedItem[i]) // button -> ON
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::realTime = true;
								 
								selectedItem[i + 1] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
							}
							else if (optionButtons[i]->id == 8 && !selectedItem[i]) // button -> OFF
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::realTime = false;

								selectedItem[i - 1] = false;
								SDL_FreeSurface(optionButtons[i - 1]->body);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
							}

							// Video Mode
							if (optionButtons[i]->id == 10 && !selectedItem[i]) // windowed
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::fullscreen = false;

								selectedItem[i + 1] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
							}
							else if (optionButtons[i]->id == 11 && !selectedItem[i]) // fullscreen
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::fullscreen = true;

								selectedItem[i - 1] = false;
								SDL_FreeSurface(optionButtons[i - 1]->body);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
							}
							// Scenes
							if (optionButtons[i]->id == 12 && !selectedItem[i])
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::sceneID = CORNELL_BOX;

								selectedItem[i + 1] = false;
								selectedItem[i + 2] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								SDL_FreeSurface(optionButtons[i + 2]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
								optionButtons[i + 2]->body = TTF_RenderText_Solid(optionButtons[i + 2]->font, optionButtons[i + 2]->name, color[0]);

								selectedItem[i + 3] = false;
								SDL_FreeSurface(optionButtons[i + 3]->body);
								optionButtons[i + 3]->body = TTF_RenderText_Solid(optionButtons[i + 3]->font, optionButtons[i + 3]->name, color[0]);
							}
							else if (optionButtons[i]->id == 13 && !selectedItem[i])
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::sceneID = ROAMING;

								selectedItem[i + 1] = false;
								selectedItem[i - 1] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								SDL_FreeSurface(optionButtons[i - 1]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);

								selectedItem[i + 2] = false;
								SDL_FreeSurface(optionButtons[i + 2]->body);
								optionButtons[i + 2]->body = TTF_RenderText_Solid(optionButtons[i + 2]->font, optionButtons[i + 2]->name, color[0]);
							}
							else if (optionButtons[i]->id == 14 && !selectedItem[i])
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::sceneID = SEA;

								selectedItem[i - 1] = false;
								selectedItem[i - 2] = false;
								SDL_FreeSurface(optionButtons[i - 1]->body);
								SDL_FreeSurface(optionButtons[i - 2]->body);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
								optionButtons[i - 2]->body = TTF_RenderText_Solid(optionButtons[i - 2]->font, optionButtons[i - 2]->name, color[0]);

								selectedItem[i + 1] = false;
								SDL_FreeSurface(optionButtons[i + 1]->body);
								optionButtons[i + 1]->body = TTF_RenderText_Solid(optionButtons[i + 1]->font, optionButtons[i + 1]->name, color[0]);
							}
							else if (optionButtons[i]->id == 15 && !selectedItem[i])
							{
								selectedItem[i] = true;
								SDL_FreeSurface(optionButtons[i]->body);
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								GlobalSettings::sceneID = ROAMING_V2;

								selectedItem[i - 1] = false;
								selectedItem[i - 2] = false;
								selectedItem[i - 3] = false;
								SDL_FreeSurface(optionButtons[i - 1]->body);
								SDL_FreeSurface(optionButtons[i - 2]->body);
								SDL_FreeSurface(optionButtons[i - 3]->body);
								optionButtons[i - 1]->body = TTF_RenderText_Solid(optionButtons[i - 1]->font, optionButtons[i - 1]->name, color[0]);
								optionButtons[i - 2]->body = TTF_RenderText_Solid(optionButtons[i - 2]->font, optionButtons[i - 2]->name, color[0]);
								optionButtons[i - 3]->body = TTF_RenderText_Solid(optionButtons[i - 3]->font, optionButtons[i - 3]->name, color[0]);
							}

							if (optionButtons[i]->id == 9)
							{
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								inOptionsMenu = false;
							}

							if (optionButtons[i]->id == 1337 && !selectedItem[i]) // open .txt file with controls
							{
								//selectedItem[i] = true;
								optionButtons[i]->body = TTF_RenderText_Solid(optionButtons[i]->font, optionButtons[i]->name, color[2]);
								ShellExecute(0, 0, "controls.txt", 0, 0, SW_SHOW);
							}
						}
					}
				}
				case SDL_MOUSEMOTION:
				{
					x = ev.motion.x;
                    y = ev.motion.y;
					if ( x >= optionButtons[8]->boundingRect.x &&
							(x <= optionButtons[8]->boundingRect.x + optionButtons[8]->boundingRect.w) &&
							 y >= optionButtons[8]->boundingRect.y &&
							(y <= optionButtons[8]->boundingRect.y + optionButtons[8]->boundingRect.h) )
					{
						if (!hoveredItem[8])
						{
							hoveredItem[8] = true;
							optionButtons[8]->body = TTF_RenderText_Solid(optionButtons[8]->font, optionButtons[8]->name, color[1]);
						}			
					}
					else
					{
						hoveredItem[8] = false;
						optionButtons[8]->body = TTF_RenderText_Solid(optionButtons[8]->font, optionButtons[8]->name, color[0]);
					}	
				}
				default:
					break;
			}
			
		}

		for (int i = 0; i < menuLabels.size(); ++i)
		{
			SDL_BlitSurface(menuLabels[i]->body, NULL, menuScreen, &menuLabels[i]->boundingRect);
		}

		for (int i = 0; i < optionButtons.size(); ++i)
		{
			SDL_BlitSurface(optionButtons[i]->body, NULL, menuScreen, &optionButtons[i]->boundingRect);
		}

		SDL_Flip(menuScreen);
	}
}

void Menu::Destroy()
{
	for (int i = 0; i < menuLabels.size(); ++i)
	{
		delete menuLabels[i];
	}

	for (int i = 0; i < optionButtons.size(); ++i)
	{
		delete optionButtons[i];
	}

	SDL_Quit();
}