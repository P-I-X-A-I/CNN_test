#include "INPUT_DATA_manager.h"


INPUT_DATA_manager::INPUT_DATA_manager()
{
	printf("\nINPUT_DATA_manager init\n");

	// initialize value
	//******************* be read by Learning manager
	DATA_X = 64;
	DATA_Y = 64;
	DATA_Z = 1;
	ANSWER_width = 26;
	DIVIDE_POINT = 100000; // train | answer 
	//******************* be read by Learning manager

	int NUM_ALPH = 26;
	int FONTS = 30;
	int VARIATION = 150;

	long imgCount = NUM_ALPH * FONTS *( DATA_X * DATA_Y * DATA_Z )*(VARIATION); // = num pixel
	long ansCount = NUM_ALPH * FONTS*(ANSWER_width)*(VARIATION);

	this->load_input_image_data("ALPHABET_DATA/alphabet_ang90_data.bin", imgCount);
	this->load_answer_data("ALPHABET_DATA/answer_117000.bin", ansCount);
}

INPUT_DATA_manager::~INPUT_DATA_manager()
{
	
}


void INPUT_DATA_manager::load_input_image_data(const char* filePath, long dataCount)
{
	// alloc memory
	if (image_ptr != nullptr) { free(image_ptr);  }

	image_ptr = (float*)malloc(dataCount * sizeof(float));

	// open file
	FILE* fp;
	errno_t err = fopen_s(&fp, filePath, "rb");

	if (err != 0) { printf("file open fail...\n"); }

	// read file
	fread_s(image_ptr,
		sizeof(float)*dataCount,
		sizeof(float),
		dataCount,
		fp);

}

void INPUT_DATA_manager::load_answer_data(const char* filePath, long dataCount)
{
	// alloc memory
	if (answer_ptr != nullptr) { free(answer_ptr);  }

	answer_ptr = (float*)malloc(dataCount*sizeof(float));

	// open file
	FILE* fp;
	errno_t err = fopen_s(&fp, filePath, "rb");

	if (err != 0) { printf("file open fail...\n");  }

	// read file
	// read file
	fread_s(answer_ptr,
		sizeof(float)*dataCount,
		sizeof(float),
		dataCount,
		fp);
}





float* INPUT_DATA_manager::get_train_image_ptr(int idx)
{
	long SKIP = DATA_X * DATA_Y * DATA_Z;

	return (image_ptr + (SKIP * idx));
}

float* INPUT_DATA_manager::get_train_answer_ptr(int idx)
{
	long SKIP = ANSWER_width;

	return (answer_ptr + (SKIP*idx));
}




float* INPUT_DATA_manager::get_check_image_ptr(int idx)
{
	
	long SKIP = DATA_X * DATA_Y * DATA_Z;

	return (image_ptr + (SKIP * (idx + DIVIDE_POINT) ));
}

float* INPUT_DATA_manager::get_check_answer_ptr(int idx)
{
	long SKIP = ANSWER_width;

	return (answer_ptr + (SKIP * (idx + DIVIDE_POINT) ));
}