#include "CHAIN_Manager_Class.h"


CHAIN_Manager_Class::CHAIN_Manager_Class()
{
	printf("CHAIN manager init\n");
	//num_OMP_thread = std::thread::hardware_concurrency();
	num_OMP_thread = NUM_CPU_CORE;
	omp_set_num_threads(NUM_CPU_CORE);
}

CHAIN_Manager_Class::~CHAIN_Manager_Class()
{

}


void CHAIN_Manager_Class::add_layer_common()
{
	for (int i = 0; i < num_OMP_thread; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count] = new LayerClass();
		//LAYER_CHAIN_OMP[i][layer_count]->num_CPU_img = 1;
		//LAYER_CHAIN_OMP[i][layer_count]->num_GPU_img = 1;
		//LAYER_CHAIN_OMP[i][layer_count]->answer_width = ANSWER_WIDTH;
		//LAYER_CHAIN_OMP[i][layer_count]->delta_limit_val = 0.1;
		//LAYER_CHAIN_OMP[i][layer_count]->LOCAL_IMG = 32;
		//LAYER_CHAIN_OMP[i][layer_count]->cl_obj = cl_manager_obj;
		LAYER_CHAIN_OMP[i][layer_count]->layer_number = layer_count;
	}
}
void CHAIN_Manager_Class::add_AFFINE(int inW, int outW, int iniWeight)
{
	// add layer common
	this->add_layer_common();

	// OMP chain
	for (int i = 0; i < num_OMP_thread; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_AFFINE(inW, outW, iniWeight);
		// affine weight is copied from save file.
	}

	// increment
	layer_count++;
}
void CHAIN_Manager_Class::add_CEMS(int inW, int mode)
{
	// init common
	this->add_layer_common();

	// OMP chain
	for (int i = 0; i < num_OMP_thread; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_CEMS(inW, mode);
	}

	// increment
	layer_count++;
}
void CHAIN_Manager_Class::add_FILTER(int inX, int inY, int inZ, int mode)
{
	// init common
	this->add_layer_common();

	// OMP chain
	for (int i = 0; i < num_OMP_thread; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_FILTER(inX, inY, inZ, mode);
	}

	// increment
	layer_count++;
}
void CHAIN_Manager_Class::add_CONVOLUTION(int inX, int inY, int inZ, int krnSize, int outZ, int weightMode)
{
	// init common
	this->add_layer_common();

	// OMP chain
	for (int i = 0; i < num_OMP_thread; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_CONVOLUTION(inX, inY, inZ, krnSize, outZ, weightMode);
		// copy weight from save file
	}

	// increment
	layer_count++;
}
void CHAIN_Manager_Class::add_MAX_POOLING(int inX, int inY, int inZ, int poolMode)
{
	// init common
	this->add_layer_common();

	// OMP chain
	for (int i = 0; i < num_OMP_thread; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_MAX_POOLING(inX, inY, inZ, poolMode);
	}

	// increment
	layer_count++;
}
void CHAIN_Manager_Class::add_DROPOUT(int inX, int inY, int inZ, float RATE, int dropMode)
{
	// init common
	this->add_layer_common();

	// OMP chain
	for (int i = 0; i < num_OMP_thread; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_DROPOUT(inX, inY, inZ, RATE, dropMode);
		// copy dropout gate from save file
	}

	// increment
	layer_count++;
}


/////////////////////////////////////////////////////
////////// LOAD FILE ////////////////////////////////
/////////////////////////////////////////////////////


void CHAIN_Manager_Class::load_file_and_create_chain_from(FILE* fp, bool isCreateChain)
{
	// load head ( layer count )
	char headNum[30] = {0};
	this->read_file_until(fp, headNum, 10); // 10 = \n

	int numLayer = atoi(headNum);
	printf("numLayer %d\n", numLayer);


	// load param of each layer
	for (int D = 0; D < numLayer; D++)
	{
		// read layer type [AFFINE, CEMS, FILTER, CONVOLUTION........]
		char layerType[32] = {0};
		this->read_file_until(fp, layerType, 10); // 10 = \n



		// read info for add layer
		/////////////////////////////////////////////////////////////////////
		if (strcmp("AFFINE", layerType) == 0)
		{
			// read 2 ( inW, outW )
			char inChar[10] = {0};
			char outChar[10] = {0};
			this->read_file_until(fp, inChar, ';');
			this->read_file_until(fp, outChar, ';');

			if (isCreateChain)
			{
				int inW = atoi(inChar);
				int outW = atoi(outChar);
				printf("AFFINE %d %d\n", inW, outW);
				// add layer
				this->add_AFFINE(inW, outW, 0);// weight mode is no meanings, weight is set by saveFile
			}

			// load weight ( only to OMP_CHAIN[0], copy to another later. )
			// [layer_count++] in add_AFFINE, so use [D]
			LAYER_CHAIN_OMP[0][D]->load_parameters_from_file(fp, layerType);

		}
		////////////////////////////////////////////////////////////
		else if (strcmp("CEMS", layerType) == 0)
		{
			// read 2 (inW, CEMSmode )
			char inChar[10] = {0};
			char modeChar[10] = { 0 };
			this->read_file_until(fp, inChar, ';');
			this->read_file_until(fp, modeChar, ';');

			if (isCreateChain)
			{
				int inW = atoi(inChar);
				int CEMSmode = atoi(modeChar);
				printf("CEMS %d %d\n", inW, CEMSmode);
				// add layer
				this->add_CEMS(inW, CEMSmode); // CEMS mode may not be meaningless, beacause no learning process
			}

			// no params to load, but skip end tag
			// [layer_count++] in add_CEMS, so use [D]
			LAYER_CHAIN_OMP[0][D]->load_parameters_from_file(fp, layerType);
		}
		//////////////////////////////////////////////////////////////////
		else if (strcmp("FILTER", layerType) == 0)
		{
			// read 4 ( inX, inY, inZ, mode )
			char inxChar[10] = {0};
			char inyChar[10] = { 0 };
			char inzChar[10] = {0};
			char modeChar[10] = {0};
			this->read_file_until(fp, inxChar, ';');
			this->read_file_until(fp, inyChar, ';');
			this->read_file_until(fp, inzChar, ';');
			this->read_file_until(fp, modeChar, ';');

			if (isCreateChain)
			{
				int inX = atoi(inxChar);
				int inY = atoi(inyChar);
				int inZ = atoi(inzChar);
				int FILTmode = atoi(modeChar);

				printf("FILTER %d %d %d %d\n", inX, inY, inZ, FILTmode);

				// add layer
				this->add_FILTER(inX, inY, inZ, FILTmode);
			}

			// no params to load, but skip end tag
			// [layer_count++] in add_FILTER, so use [D]
			LAYER_CHAIN_OMP[0][D]->load_parameters_from_file(fp, layerType);

		}
		else if (strcmp("CONVOLUTION", layerType) == 0)
		{
			// read 5 (inX, inY, inZ, kSize, outZ )
			char inxChar[10] = {0};
			char inyChar[10] = {0};
			char inzChar[10] = {0};
			char ksizeChar[10] = {0};
			char outzChar[10] = {0};
			this->read_file_until(fp, inxChar, ';');
			this->read_file_until(fp, inyChar, ';');
			this->read_file_until(fp, inzChar, ';');
			this->read_file_until(fp, ksizeChar, ';');
			this->read_file_until(fp, outzChar, ';');


			if (isCreateChain)
			{
				int inX = atoi(inxChar);
				int inY = atoi(inyChar);
				int inZ = atoi(inzChar);
				int kSize = atoi(ksizeChar);
				int outZ = atoi(outzChar);

				printf("CONV %d %d %d %d %d\n", inX, inY, inZ, kSize, outZ);

				// add layer
				this->add_CONVOLUTION(inX, inY, inZ, kSize, outZ, 0); // weight mode is meaningless, weights are set by saveFile
			}

			// read bias, kernel
			// [layer_count++] in add_CONVOLUTION, so use [D]
			LAYER_CHAIN_OMP[0][D]->load_parameters_from_file(fp, layerType);
		}
		else if (strcmp("POOLING", layerType) == 0)
		{
			// read 4 (inX, inY, inZ, poolMode )
			char inxChar[10] = { 0 };
			char inyChar[10] = {0};
			char inzChar[10] = {0};
			char modeChar[10] = {0};
			this->read_file_until(fp, inxChar, ';');
			this->read_file_until(fp, inyChar, ';');
			this->read_file_until(fp, inzChar, ';');
			this->read_file_until(fp, modeChar, ';');


			if (isCreateChain)
			{
				int inX = atoi(inxChar);
				int inY = atoi(inyChar);
				int inZ = atoi(inzChar);
				int POOLmode = atoi(modeChar);

				printf("POOL %d %d %d %d\n", inX, inY, inZ, POOLmode);

				// add layer
				this->add_MAX_POOLING(inX, inY, inZ, POOLmode);
			}

			// no params to load, but skip end tag
			// [layer_count++] in add_POOLING, so use [D]
			LAYER_CHAIN_OMP[0][D]->load_parameters_from_file(fp, layerType);
		}
		else if (strcmp("DROPOUT", layerType) == 0)
		{
			// read 3 (inX, inY, inZ )
			char inxChar[10] = {0};
			char inyChar[10] = {0};
			char inzChar[10] = { 0 };
			this->read_file_until(fp, inxChar, ';');
			this->read_file_until(fp, inyChar, ';');
			this->read_file_until(fp, inzChar, ';');

			if (isCreateChain)
			{
				int inX = atoi(inxChar);
				int inY = atoi(inyChar);
				int inZ = atoi(inzChar);

				// add layer
				this->add_DROPOUT(inX, inY, inZ, 0.1, 1); // RATE is meaningless, 1 = ACT_USE_MODE
			}

			// load saved dropout-gate.
			// [layer_count++] in add_DROPOUT, so use [D]
			LAYER_CHAIN_OMP[0][D]->load_parameters_from_file(fp, layerType);
		}
		

		//LAYER_CHAIN_OMP[0][D]->load_parameters_from_file(fp);
	}



	// copy weight to another OMP Chain
}




// utility
void CHAIN_Manager_Class::read_file_until(FILE* fromFp, char* readBuf, char untilCode)
{
	char readChar = 0;
	char* writeBuf = readBuf;

	while (1)
	{
		// read one character
		fread_s(&readChar, sizeof(readChar), sizeof(char), 1, fromFp);

		// check until code
		if (readChar == untilCode)
		{
			break;
		}

		// write to buf
		*writeBuf = readChar;
		writeBuf++;
	}
}