#pragma once
#include "framework.h"


class INPUT_DATA_manager
{
public:

	INPUT_DATA_manager();
	~INPUT_DATA_manager();

	int DATA_X = 64;
	int DATA_Y = 64;
	int DATA_Z = 1;
	int ANSWER_width = 26;
	int DIVIDE_POINT = 100;


	float* image_ptr = nullptr;
	float* answer_ptr = nullptr;


	void load_input_image_data(const char* filePath, long dataCount);
	void load_answer_data(const char* filePath, long dataCount);


	float* get_train_image_ptr(int idx);
	float* get_train_answer_ptr(int idx);

	float* get_check_image_ptr(int idx);
	float* get_check_answer_ptr(int idx);
};

