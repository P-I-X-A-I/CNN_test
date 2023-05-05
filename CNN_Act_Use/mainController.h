#pragma once
#include "framework.h"

#include "LayerClass.h"
#include "INPUT_DATA_manager.h"

#include "OpenGL_Manager_Class.h"
#include "Window_Manager_Class.h"
#include "GUI_Manager_Class.h"

#include "CHAIN_Manager_Class.h"


class mainController
{
public:

	mainController();
	~mainController();

	const char* alp[26];

	OpenGL_Manager_Class* gl_manager_obj;
	Window_Manager_Class* window_manager_obj;
	GUI_Manager_Class* gui_manager_obj;

	CHAIN_Manager_Class* CHAIN_manager_obj;
	INPUT_DATA_manager* INPUT_data_obj;

	HWND base_window_obj;
	HWND gl_view_obj;

	void setup_OpenGL();
};

