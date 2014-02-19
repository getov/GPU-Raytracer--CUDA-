#pragma once

#include <SDL.h>
#include <SDL_ttf.h>
#include <vector>
#include <unordered_map>

class Button
{
private:
	int _fontSize;

public:
	const char* name;
	TTF_Font* font;
	SDL_Color color;
	SDL_Surface* body;
	SDL_Rect boundingRect;
	int id;

	Button(const char* buttonName, const SDL_Color& buttonColor, const int& buttonID, int fontSize = 20);
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

	/**
	* @brief - initialize the SDL_Color array
	* with different colors for different states
	* - gray         - default color
	* - lighter gray - for hovered button
	* - green        - for selected button
	* - white		 - for menu items
	*/
	void setTextColor();

	/**
	* @brief - Render the menu options on the screen
	*/
	void renderMenuLabels();

	/**
	* @brief - Render the option buttons on the screen
	*/
	void renderOptionButtons();

	/**
	* @brief - Handles the mouse motion and mouse
	* click events with the SDL_PollEvent system
	*/
	void handleMouseEvents();

	void Destroy();
};