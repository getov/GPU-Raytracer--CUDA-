#pragma once

#include <SDL.h>
#include <SDL_ttf.h>
#include <vector>
#include <unordered_map>

class Button
{
public:
	const char* name;
	TTF_Font* font;
	SDL_Color color;
	SDL_Surface* body;
	SDL_Rect boundingRect;
	int id;

	Button(const char* buttonName, const SDL_Color& buttonColor, const int& buttonID);
};

class Menu
{
private:
	Menu(const Menu& ev);
	Menu& operator=(const Menu& ev);

	SDL_Surface* menuScreen;

	const char* appName;

	std::vector<bool> selectedItem;
	std::vector<bool> hoveredItem;

	std::vector<Button*> menuLabels;
	std::vector<Button*> optionButtons;

	SDL_Color color[4];

public:
	Menu(const char* name);

	void setTextColor();

	void renderMenuLabels();
	void renderOptionButtons();

	void handleMouseEvents();

	void Destroy();
};