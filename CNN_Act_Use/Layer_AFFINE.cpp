#include "LayerClass.h"


void LayerClass::init_as_AFFINE(int inW, int outW, int iniWeight)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "AFFINE");
	
	input_dim[0] = inW;
	input_dim[1] = 1;
	input_dim[2] = 1;
	output_dim[0] = outW;
	output_dim[1] = 1;
	output_dim[2] = 1;

	AFFINE_weight_mode = iniWeight; // 0 Xavier, 1 He, 2 Const

	const char* tempStr[3];
	tempStr[0] = "Xavier";
	tempStr[1] = "He";
	tempStr[2] = "Const";

	printf("init Layer as [%s (%s)]\n", layer_type_str, tempStr[AFFINE_weight_mode]);

	this->alloc_AFFINE_CPU_memory();
}


////////////////////////////////////////////////
///////// CPU MEMORY SETUP /////////////////////
////////////////////////////////////////////////



void LayerClass::alloc_AFFINE_CPU_memory()
{
	long dataSize;
	
	// input data ( set by prev )

	// output data
	dataSize = sizeof(float) * output_dim[0];
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	// back-in data ( set by prev )

	// back-out data
	dataSize = sizeof(float)*input_dim[0];
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	// unique variables *********************

	//bias
	dataSize = sizeof(float)*output_dim[0];
	AFFINE_bias_ptr = (float*)malloc(dataSize);
	memset(AFFINE_bias_ptr, 0, dataSize);

	//weight
	dataSize = sizeof(float)*input_dim[0] * output_dim[0];
	AFFINE_weight_ptr = (float*)malloc(dataSize);
	memset(AFFINE_weight_ptr, 0, dataSize);

	// set num saved parameter///////////////////////////////////////
	num_saved_param = output_dim[0] + (input_dim[0] * output_dim[0]); 
	/////////////////////////////////////////////////////////////////

	// bias delta sum
	dataSize = sizeof(float) * output_dim[0];
	AFFINE_bias_delta_sum = (float*)malloc(dataSize);
	memset(AFFINE_bias_delta_sum, 0, dataSize);

	// weight delta sum
	dataSize = sizeof(float) * input_dim[0] * output_dim[0];
	AFFINE_weight_delta_sum = (float*)malloc(dataSize);
	memset(AFFINE_weight_delta_sum, 0, dataSize);

	// bias velocity
	dataSize = sizeof(float) * output_dim[0];
	AFFINE_bias_velocity = (float*)malloc(dataSize);
	memset(AFFINE_bias_velocity, 0, dataSize);

	// weight velocity
	dataSize = sizeof(float) * input_dim[0] * output_dim[0];
	AFFINE_weight_velocity = (float*)malloc(dataSize);
	memset(AFFINE_weight_velocity, 0, dataSize);

	// bias ADAG history
	dataSize = sizeof(float)*output_dim[0] * ADAG_HISTORY;
	AFFINE_bias_ADAG_history = (float*)malloc(dataSize);
	
	for (int i = 0; i < (output_dim[0] * ADAG_HISTORY); i++)
	{
		*(AFFINE_bias_ADAG_history + i) = 1.0; // init by 1.0
	}

	// weight ADAG_history
	dataSize = sizeof(float)*input_dim[0] * output_dim[0] * ADAG_HISTORY;
	AFFINE_weight_ADAG_history = (float*)malloc(dataSize);

	for (int i = 0; i < (input_dim[0] * output_dim[0] * ADAG_HISTORY); i++)
	{
		*(AFFINE_weight_ADAG_history + i) = 1.0; // init by 1.0
	}


	///////////////////////////////////
	this->generate_AFFINE_initial_HX_weight();

}



void LayerClass::generate_AFFINE_initial_HX_weight()
{
	//
	float DEVI_weight = 1.0;
	float DEVI_bias = 1.0;
	float n_weight = (float)input_dim[0];
	//float n_bias = (float)output_dim[0];

	switch (AFFINE_weight_mode)
	{
	case 0: // xavier
		DEVI_weight = sqrt(1.0 / n_weight);
		//DEVI_bias = sqrt(1.0 / n_bias);
		break;

	case 1: // He
		DEVI_weight = sqrt(2.0 / n_weight);
		//DEVI_bias = sqrt(2.0 / n_bias);
		break;

	case 2:
		DEVI_weight = 0.2;
		//DEVI_bias = 0.2;
		break;
	}

	// C++ random
	std::random_device rDevice; // normal random device
	std::mt19937 mersenne(rDevice()); // 32bit Mersenne Twister
	std::normal_distribution<> norm_dist_weight(0.0, DEVI_weight); // 
	//std::normal_distribution<> norm_dist_bias(0.0, DEVI_bias); // 

	// write weight
	int acc = 0;
	for (int n = 0; n < output_dim[0]; n++)
	{
		for (int w = 0; w < input_dim[0]; w++)
		{
			*(AFFINE_weight_ptr + acc) = norm_dist_weight(mersenne);
			acc++;
		}
	}

	// write bias
	for (int n = 0; n < output_dim[0]; n++)
	{
		//*(AFFINE_bias_ptr + n) = norm_dist_bias(mersenne);
		*(AFFINE_bias_ptr + n) = 0.0; // initial bias = 0.0
	}
}


/////////////////////////////////////////////////////////
/////////// OPEN CL SETUP ///////////////////////////////
/////////////////////////////////////////////////////////


void LayerClass::setup_AFFINE_cl_mem()
{
	long dataSize;

	// input mem ( set by prev )

	// output mem
	dataSize = sizeof(float) * output_dim[0] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_output_data, dataSize, CL_MEM_READ_WRITE);

	// back-in mem ( set by prev )

	// back-out mem
	dataSize = sizeof(float) * input_dim[0] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_back_out, dataSize, CL_MEM_READ_WRITE);

	// bias
	dataSize = sizeof(float) * output_dim[0]; // no image layer
	cl_obj->create_mem_util(DEVICE_id, &mem_AFFINE_bias, dataSize, CL_MEM_READ_ONLY);
	// copy bias value to cl_mem
	cl_obj->update_mem_contents_util(DEVICE_id, &mem_AFFINE_bias, AFFINE_bias_ptr, dataSize);


	// weight
	dataSize = sizeof(float) * input_dim[0] * output_dim[0]; // no image layer
	cl_obj->create_mem_util(DEVICE_id, &mem_AFFINE_weight, dataSize, CL_MEM_READ_ONLY);
	// copy weight value to cl_mem *>*>*>*>*>*>*>**>*>*>*>
	cl_obj->update_mem_contents_util(DEVICE_id, &mem_AFFINE_weight, AFFINE_weight_ptr, dataSize);


	// bias delta sum
	dataSize = sizeof(float) * output_dim[0]; // no image layer
	cl_obj->create_mem_util(DEVICE_id, &mem_AFFINE_bias_delta_sum, dataSize, CL_MEM_WRITE_ONLY);

	// bias delta weight
	dataSize = sizeof(float)*input_dim[0] * output_dim[0]; // no image layer
	cl_obj->create_mem_util(DEVICE_id, &mem_AFFINE_weight_delta_sum, dataSize, CL_MEM_WRITE_ONLY);

}



void LayerClass::setup_AFFINE_cl_kernel()
{
	// kernel
	cl_obj->create_kernel_util(DEVICE_id, &krn_AFFINE, "affine_weight");
	cl_obj->set_kernel_arg_mem(&krn_AFFINE, 0, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE, 1, &mem_AFFINE_weight);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE, 2, &mem_AFFINE_bias);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE, 3, &mem_output_data);
	cl_obj->set_kernel_arg_int_val(&krn_AFFINE, 4, input_dim[0]);

	// kernel ( back )
	cl_obj->create_kernel_util(DEVICE_id, &krn_AFFINE_back_biasSum, "affine_back_bias_delta_sum");
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_biasSum, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_biasSum, 1, &mem_AFFINE_bias_delta_sum);
	cl_obj->set_kernel_arg_int_val(&krn_AFFINE_back_biasSum, 2, num_GPU_img);

	cl_obj->create_kernel_util(DEVICE_id, &krn_AFFINE_back_weightSum, "affine_back_weight_delta_sum");
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_weightSum, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_weightSum, 1, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_weightSum, 2, &mem_AFFINE_weight_delta_sum);
	cl_obj->set_kernel_arg_int_val(&krn_AFFINE_back_weightSum, 3, num_GPU_img);

	cl_obj->create_kernel_util(DEVICE_id, &krn_AFFINE_back_out, "affine_back_out");
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_out, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_out, 1, &mem_AFFINE_weight);
	cl_obj->set_kernel_arg_mem(&krn_AFFINE_back_out, 2, &mem_back_out);
	cl_obj->set_kernel_arg_int_val(&krn_AFFINE_back_out, 3, output_dim[0]);

}


////////////////////////////////////////////////
/////////// CPU PROCESS ////////////////////////
////////////////////////////////////////////////



void LayerClass::learn_CPU_AFFINE(float* inPtr, float* asnPtr)
{
	input_data_ptr = inPtr;

	// in-data * weight -> output
	for (int n = 0; n < output_dim[0]; n++)
	{
		float SUM = 0.0;
		float* weightHead = AFFINE_weight_ptr + (input_dim[0] * n);

		for (int w = 0; w < input_dim[0]; w++)
		{
			float inVal = *(input_data_ptr + w);
			float weiVal = *(weightHead + w);

			SUM += inVal * weiVal;
		}

		// add bias
		SUM += *(AFFINE_bias_ptr + n);

		// write out
		*(output_data_ptr + n) = SUM;
	}
}


void LayerClass::back_propagation_AFFINE(float* backInPtr)
{
	back_in_ptr = backInPtr;

	// bias delta is back-in value itself
	for (int i = 0; i < output_dim[0]; i++)
	{
		float backVal = *(back_in_ptr + i);
		*(AFFINE_bias_delta_sum + i) += backVal; // sumup through 1 loop
	}


	// weight delta (back-in) * (prev_input)
	for (int n = 0; n < output_dim[0]; n++)
	{
		float backVal = *(back_in_ptr + n);

		//weight
		float* weightSumHead = AFFINE_weight_delta_sum + (input_dim[0] * n);

		for (int w = 0; w < input_dim[0]; w++)
		{
			float prevVal = *(input_data_ptr + w);

			*(weightSumHead + w) += backVal * prevVal;
		}
	}


	// back-out value
	for (int w = 0; w < input_dim[0]; w++)
	{
		float backSum = 0.0;

		for (int n = 0; n < output_dim[0]; n++)
		{
			float backInVal = *(back_in_ptr + n);
			float weiVal = *(AFFINE_weight_ptr + (n * input_dim[0]) + w);

			backSum += backInVal * weiVal;
		}
	
		// write out
		*(back_out_ptr + w) = backSum;
	}

}

//////////////////////////////////////////////////
///////// GPU PROCESS ////////////////////////////
//////////////////////////////////////////////////


void LayerClass::enqueue_forward_kernel_AFFINE()
{

	int evenOdd = 1;
	evenOdd = this->find_divisor(output_dim[0], 8);


	// outW * image parallel
	cl_int err;
	size_t off2D[2] = { 0, 0 };
	size_t work2D[2] = { output_dim[0], num_GPU_img };
	size_t local2D[2] = { evenOdd, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	err = clEnqueueNDRangeKernel(CMQ,
		krn_AFFINE,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_AFFINE] fail...%d\n", err); }
}


void LayerClass::enqueue_back_kernel_AFFINE()
{

	int evenOdd_in = 1;
	int evenOdd_out = 1;

	evenOdd_in = this->find_divisor(input_dim[0], 8);
	evenOdd_out = this->find_divisor(output_dim[0], 8);


	cl_int err;
	size_t off2D[2] = { 0, 0 };
	size_t work2D[2];
	size_t local2D[2];

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	// back bias ( outW parallel )
	work2D[0] = output_dim[0];
	work2D[1] = 1;
	local2D[0] = evenOdd_out;
	local2D[1] = 1;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_AFFINE_back_biasSum,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_AFFINE_back_biasSum] fail...%d\n", err); }

	 
	// back weight ( inW * outW parallel )
	work2D[0] = input_dim[0];
	work2D[1] = output_dim[0];
	local2D[0] = evenOdd_in;
	local2D[1] = evenOdd_out;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_AFFINE_back_weightSum,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_AFFINE_back_weightSum] fail...%d\n", err); }


	// back out (inW * image parallel )
	work2D[0] = input_dim[0];
	work2D[1] = num_GPU_img;
	local2D[0] = evenOdd_in;
	local2D[1] = LOCAL_IMG;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_AFFINE_back_out,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_AFFINE_back_out] fail...%d\n", err); }

}


void LayerClass::enqueue_read_back_kernel_AFFINE()
{
	// read back weight & bias sum
	cl_int err;
	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	err = clEnqueueReadBuffer(CMQ,
		mem_AFFINE_bias_delta_sum,
		CL_FALSE, 0,
		sizeof(float)*output_dim[0],
		AFFINE_bias_delta_sum,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("mem read-back[affine bias-delta] fail...%d\n", err); }

	//////////////////////////////////////////////

	err = clEnqueueReadBuffer(CMQ,
		mem_AFFINE_weight_delta_sum,
		CL_FALSE, 0,
		sizeof(float)*input_dim[0] * output_dim[0],
		AFFINE_weight_delta_sum,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("mem read-back[affine weight delta]fail...%d\n", err); }
}