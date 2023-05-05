#pragma once
#include "framework.h"

#include "LayerClass.h"

#include <omp.h>
#include <thread>


#define NUM_CPU_CORE 1

class CHAIN_Manager_Class
{
public:

	CHAIN_Manager_Class();
	~CHAIN_Manager_Class();


	int num_OMP_thread = 1;//////////////////////////////////////
	int layer_count = 0;

	// LayerChain ****************************
	LayerClass* LAYER_CHAIN_OMP[NUM_CPU_CORE][128]; // max deepness 128



	// setup layer
	void add_layer_common();
	void add_AFFINE(int inW, int outW, int iniWeight);
	void add_CEMS(int inW, int mode);
	void add_FILTER(int inX, int inY, int inZ, int mode);
	void add_CONVOLUTION(int inX, int inY, int inZ, int krnSize, int outZ, int weightMode);
	void add_MAX_POOLING(int inX, int inY, int inZ, int poolMode);
	void add_DROPOUT(int inX, int inY, int inZ, float RATE, int dropMode);


	// load
	void load_file_and_create_chain_from(FILE* fp, bool isCreateChain );
	void read_file_until(FILE* fromFp, char* readBuf, char untilCode);
};

