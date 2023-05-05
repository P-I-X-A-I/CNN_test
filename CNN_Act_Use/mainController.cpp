#include "mainController.h"


mainController::mainController()
{
	printf("mainController init\n");

	// input data 
	INPUT_data_obj = new INPUT_DATA_manager();

	alp[0] = "A";
	alp[1] = "B";
	alp[2] = "C";
	alp[3] = "D";
	alp[4] = "E";
	alp[5] = "F";
	alp[6] = "G";
	alp[7] = "H";
	alp[8] = "I";
	alp[9] = "J";
	alp[10] = "K";
	alp[11] = "L";
	alp[12] = "M";
	alp[13] = "N";
	alp[14] = "O";
	alp[15] = "P";
	alp[16] = "Q";
	alp[17] = "R";
	alp[18] = "S";
	alp[19] = "T";
	alp[20] = "U";
	alp[21] = "V";
	alp[22] = "W";
	alp[23] = "X";
	alp[24] = "Y";
	alp[25] = "Z";


	//init chain manager
	CHAIN_manager_obj = new CHAIN_Manager_Class();



	// load from savefile
	FILE* saveFp;
	errno_t err = fopen_s(&saveFp, "SAVEFILE/ang90-1000", "r");
	
	if (err != 0)
	{
		printf("Open save file fail.\n");
	}
	else
	{
		CHAIN_manager_obj->load_file_and_create_chain_from(saveFp, true);
	
	}


	// load from save file

	this->setup_OpenGL();



	// learn and check result
	float ACCU_COUNTER[26] = { 0.0 };
	for (int TIMES = 0; TIMES < 30; TIMES++)
	{
		for (int alphabet = 0; alphabet < 26; alphabet++)
		{
			ACCU_COUNTER[alphabet] = 0.0;

			for (int f = 0; f < 30; f++)
			{

				// get input data
				int ID = (TIMES * 30 * 26) + (f * 26) + alphabet;
				float* inPtr = INPUT_data_obj->get_train_image_ptr(ID);


				// learn process
				int LAYER_COUNT = CHAIN_manager_obj->layer_count;

				for (int D = 0; D < LAYER_COUNT - 1; D++) // learn -1 ( except for last CEMS )
				{
					LayerClass* tempLayer = CHAIN_manager_obj->LAYER_CHAIN_OMP[0][D];
					tempLayer->learn_CPU(inPtr, nullptr);

					// updata next input
					inPtr = tempLayer->output_data_ptr;
				}


				// get result
				int layerCount = CHAIN_manager_obj->layer_count;
				LayerClass* resultLayer = CHAIN_manager_obj->LAYER_CHAIN_OMP[0][layerCount - 2]; // softmax layer
				float* resultPtr = resultLayer->output_data_ptr;

				float maxVal = 0.0;
				int maxID = 255;

				for (int r = 0; r < 26; r++)
				{
					float resVal = *(resultPtr + r);
					//if (alphabet == r) { printf("%d/", (int)(resVal * 100.0)); }
					if (resVal > maxVal && resVal >= 0.85)
					{
						maxVal = resVal;
						maxID = r;
					}
				}

				if (maxID == alphabet)
				{
					ACCU_COUNTER[alphabet] += 1.0;
				}

				// print alphabet
				if (maxID != 255)
				{
					printf("%s", alp[maxID]);
				}
				else
				{
					printf("_");
				}


			}// for f
			printf("  [%1.4f]\n", ACCU_COUNTER[alphabet] / 30.0);
		}// for alh
		printf("**************************\n");
	}

}



mainController::~mainController()
{

}


void mainController::setup_OpenGL()
{
	gl_manager_obj = new OpenGL_Manager_Class();
	window_manager_obj = new Window_Manager_Class();
	window_manager_obj->setMainController(this);
	gui_manager_obj = new GUI_Manager_Class();

	// create dummy window
	HWND dummyWnd = window_manager_obj->create_borderless_window(L"dummy");
}