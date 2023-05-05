#include "LayerClass.h"

void LayerClass::init_as_FILTER(int inX, int inY, int inZ, int mode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "FILTER");

	input_dim[0] = output_dim[0] = inX;
	input_dim[1] = output_dim[1] = inY;
	input_dim[2] = output_dim[2] = inZ;

	FILTER_mode = mode;

	const char* tempStr[5];
	tempStr[0] = "batch-norm";
	tempStr[1] = "softmax";
	tempStr[2] = "ReLU";
	tempStr[3] = "c";
	tempStr[4] = "d";

	printf("init Layer as [%s (%s)]\n", layer_type_str, tempStr[mode]);

	this->alloc_FILTER_CPU_memory();
}


void LayerClass::alloc_FILTER_CPU_memory()
{
	long dataSize;
	long TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	// in data ( set by prev )

	// out data
	dataSize = sizeof(float) * TOTAL;
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	// back-in ( set by prev )

	// back-out
	dataSize = sizeof(float) * TOTAL;
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	///////////////////////////////////////

	// scale ( batch norm )
	dataSize = sizeof(float)*output_dim[2];
	FILTER_batchN_scale_ptr = (float*)malloc(dataSize);
	for (int i = 0; i < output_dim[2]; i++)
	{
		*(FILTER_batchN_scale_ptr + i) = 1.0; // init by 1.0
	}

	// bias ( batch norm )
	dataSize = sizeof(float)*output_dim[2];
	FILTER_batchN_bias_ptr = (float*)malloc(dataSize);
	for (int i = 0; i < output_dim[2]; i++)
	{
		*(FILTER_batchN_bias_ptr + i) = 0.0; // init by 0.0
	}

	// hold mediate value ( input to sacale ) for back-propagation
	dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2];
	FILTER_batchN_hold_midVal = (float*)malloc(dataSize);
	memset(FILTER_batchN_hold_midVal, 0, dataSize);

	// scale delta sum
	dataSize = sizeof(float) * output_dim[2];
	FILTER_batchN_scale_delta_sum = (float*)malloc(dataSize);
	for (int i = 0; i < output_dim[2]; i++)
	{
		*(FILTER_batchN_scale_delta_sum + i) = 0.0;
	}

	// bias delta sum
	dataSize = sizeof(float) * output_dim[2];
	FILTER_batchN_bias_delta_sum = (float*)malloc(dataSize);
	for (int i = 0; i < output_dim[2]; i++)
	{
		*(FILTER_batchN_bias_delta_sum + i) = 0.0;
	}

	// scale velocity
	dataSize = sizeof(float) * output_dim[2];
	FILTER_batchN_scale_velocity = (float*)malloc(dataSize);
	for (int i = 0; i < output_dim[2]; i++)
	{
		*(FILTER_batchN_scale_velocity + i) = 0.0;
	}

	// bias velocity
	dataSize = sizeof(float) * output_dim[2];
	FILTER_batchN_bias_velocity = (float*)malloc(dataSize);
	for (int i = 0; i < output_dim[2]; i++)
	{
		*(FILTER_batchN_bias_velocity + i) = 0.0;
	}


	// scale ADAG history
	dataSize = sizeof(float) * output_dim[2] * ADAG_HISTORY;
	FILTER_batchN_scale_ADAG_history = (float*)malloc(dataSize);
	for (int i = 0; i < (output_dim[2] * ADAG_HISTORY); i++)
	{
		*(FILTER_batchN_scale_ADAG_history + i) = 1.0;// init by 1.0
	}

	// bias ADAG history
	dataSize = sizeof(float) * output_dim[2] * ADAG_HISTORY;
	FILTER_batchN_bias_ADAG_history = (float*)malloc(dataSize);
	for (int i = 0; i < (output_dim[2] * ADAG_HISTORY); i++)
	{
		*(FILTER_batchN_bias_ADAG_history + i) = 1.0; // init by 1.0
	}


	//////////////////////////////////////////////

	// hold deviation ( batch norm )
	dataSize = sizeof(float);
	FILTER_hold_DEVI_ptr = (float*)malloc(dataSize);
	*FILTER_hold_DEVI_ptr = 0.0;

	// hold deviation each ( batch norm ) // test
	dataSize = sizeof(float)*output_dim[2];
	FILTER_hold_DEVI_each_ptr = (float*)malloc(dataSize);
	memset(FILTER_hold_DEVI_each_ptr, 0, dataSize);

	// 

	///////////////////////////////////////

	// hold exp & expSum
	dataSize = sizeof(float) * TOTAL;
	FILTER_hold_exp = (float*)malloc(dataSize);
	memset(FILTER_hold_exp, 0, dataSize);

	dataSize = sizeof(float) * 1;
	FILTER_hold_expSum = (float*)malloc(dataSize);
	*FILTER_hold_expSum = 0.0;


	// set num saved parameter///////////////////////////////////////
	num_saved_param = 0;
	/////////////////////////////////////////////////////////////////
}




void LayerClass::setup_FILTER_cl_mem()
{
	long dataSize;
	long TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	// input mem ( set by prev )

	// output mem
	dataSize = sizeof(float) * TOTAL * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_output_data, dataSize, CL_MEM_READ_WRITE);

	// back-in mem ( set by prev )

	// back-out mem
	dataSize = sizeof(float) * TOTAL * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_back_out, dataSize, CL_MEM_READ_WRITE);
	
	///// BATCH NORM ///////////

	// batch norm scale
	dataSize = sizeof(float) * output_dim[2]; // no image layer
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_batchN_scale, dataSize, CL_MEM_READ_ONLY);
	// update mem contents
	cl_obj->update_mem_contents_util(DEVICE_id, &mem_FILTER_batchN_scale, FILTER_batchN_scale_ptr, dataSize);

	// batch norm bias
	dataSize = sizeof(float) * output_dim[2]; // no image layer
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_batchN_bias, dataSize, CL_MEM_READ_ONLY);
	// update mem contents
	cl_obj->update_mem_contents_util(DEVICE_id, &mem_FILTER_batchN_bias, FILTER_batchN_bias_ptr, dataSize);

	// hold midVal for back P
	dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_batchN_hold_midVal, dataSize, CL_MEM_READ_WRITE);

	// scale delta each
	dataSize = sizeof(float)*output_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_batchN_scale_delta_each, dataSize, CL_MEM_READ_WRITE);

	// bias delta each
	dataSize = sizeof(float)*output_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_batchN_bias_delta_each, dataSize, CL_MEM_READ_WRITE);

	// scale delta sum ( no image layer )
	dataSize = sizeof(float)*output_dim[2];
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_batchN_scale_delta_sum, dataSize, CL_MEM_READ_WRITE);

	// bias delta sum ( no image layer )
	dataSize = sizeof(float)*output_dim[2];
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_batchN_bias_delta_sum, dataSize, CL_MEM_READ_WRITE);


	/////////////////////////////////////////////////////////////////////////////////////////


	// batch norm back P
	dataSize = sizeof(float) * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_hold_DEVI, dataSize, CL_MEM_READ_WRITE);

	dataSize = sizeof(float)*output_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_hold_DEVI_each, dataSize, CL_MEM_READ_WRITE);

	// // temp for calculation
	dataSize = sizeof(float) * TOTAL * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_temp, dataSize, CL_MEM_READ_WRITE);

	//// SOFT MAX /////////////
	dataSize = sizeof(float) * TOTAL * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_hold_exp, dataSize, CL_MEM_READ_WRITE);

	dataSize = sizeof(float) * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_FILTER_hold_expSum, dataSize, CL_MEM_READ_WRITE);


}




void LayerClass::setup_FILTER_cl_kernel()
{
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	switch (FILTER_mode)
	{
	case 0: // batch norm
		/*
		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_batch_norm, "filter_batch_norm");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm, 0, &mem_input_data);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm, 1, &mem_FILTER_hold_DEVI);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm, 2, &mem_output_data);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_batch_norm, 3, TOTAL);
		*/
		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_batch_norm_each_z, "filter_batch_norm_each_z");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm_each_z, 0, &mem_input_data);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm_each_z, 1, &mem_FILTER_hold_DEVI_each);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm_each_z, 2, &mem_FILTER_batchN_hold_midVal);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm_each_z, 3, &mem_FILTER_batchN_scale);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm_each_z, 4, &mem_FILTER_batchN_bias);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_batch_norm_each_z, 5, &mem_output_data);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_batch_norm_each_z, 6, input_dim[0]);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_batch_norm_each_z, 7, input_dim[1]);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_batch_norm_each_z, 8, input_dim[2]);

		/*
		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_back_batch_norm, "filter_back_batch_norm");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm, 0, &mem_back_in);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm, 1, &mem_output_data);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm, 2, &mem_FILTER_hold_DEVI);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm, 3, &mem_back_out);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm, 4, &mem_FILTER_temp);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_back_batch_norm, 5, TOTAL);
		*/

		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_back_batch_norm_each_z, "filter_back_batch_norm_each_z");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 0, &mem_back_in);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 1, &mem_output_data);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 2, &mem_FILTER_batchN_scale);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 3, &mem_FILTER_batchN_bias_delta_each);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 4, &mem_FILTER_batchN_scale_delta_each);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 5, &mem_FILTER_batchN_hold_midVal);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 6, &mem_FILTER_hold_DEVI_each);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 7, &mem_back_out);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_each_z, 8, &mem_FILTER_temp);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_back_batch_norm_each_z, 9, input_dim[0]);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_back_batch_norm_each_z, 10, input_dim[1]);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_back_batch_norm_each_z, 11, input_dim[2]);


		// sum delta
		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_back_batch_norm_sumDelta, "filter_batch_norm_sum_delta");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_sumDelta, 0, &mem_FILTER_batchN_bias_delta_each);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_sumDelta, 1, &mem_FILTER_batchN_bias_delta_sum);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_sumDelta, 2, &mem_FILTER_batchN_scale_delta_each);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_batch_norm_sumDelta, 3, &mem_FILTER_batchN_scale_delta_sum);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_back_batch_norm_sumDelta, 4, num_GPU_img);
		break;

	case 1: // softmax
		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_softmax, "filter_softmax");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_softmax, 0, &mem_input_data);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_softmax, 1, &mem_FILTER_hold_exp);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_softmax, 2, &mem_FILTER_hold_expSum);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_softmax, 3, &mem_output_data);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_softmax, 4, TOTAL);

		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_back_softmax, "filter_back_softmax");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_softmax, 0, &mem_back_in);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_softmax, 1, &mem_FILTER_hold_exp);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_softmax, 2, &mem_FILTER_hold_expSum);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_softmax, 3, &mem_back_out);
		cl_obj->set_kernel_arg_int_val(&krn_FILTER_back_softmax, 4, TOTAL);
		break;

	case 2: // ReLU
		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_ReLU, "filter_ReLU");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_ReLU, 0, &mem_input_data);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_ReLU, 1, &mem_output_data);

		cl_obj->create_kernel_util(DEVICE_id, &krn_FILTER_back_ReLU, "filter_back_ReLU");
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_ReLU, 0, &mem_back_in);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_ReLU, 1, &mem_output_data);
		cl_obj->set_kernel_arg_mem(&krn_FILTER_back_ReLU, 2, &mem_back_out);

		break;
	case 3:
		break;
	}
}



void LayerClass::learn_CPU_FILTER(float* inPtr, float* ansPtr)
{
	// set input ptr
	input_data_ptr = inPtr;

	switch (FILTER_mode)
	{
	case 0: // batch norm
		this->FILTER_batch_norm(inPtr);
		break;

	case 1: // soft max
		this->FILTER_softmax(inPtr);
		break;

	case 2: // ReLU
		this->FILTER_ReLU(inPtr);
		break;

	case 3:
		break;
	}
}





void LayerClass::FILTER_batch_norm(float* inPtr)
{
	/*
	// calc average
	float ave = 0.0;
	float S = 0.0;

	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	// sum up input data
	for (int i = 0; i < TOTAL; i++)
	{
		S += *(input_data_ptr + i);
	}

	ave = S / (float)TOTAL;


	// calc deviation
	float DEVI = 0.0;

	for (int i = 0; i < TOTAL; i++)
	{
		float TEMP = (*(input_data_ptr + i) - ave);
		DEVI += (TEMP*TEMP);
	}

	DEVI /= (float)TOTAL;

	// hold deviation for backP
	*FILTER_hold_DEVI_ptr = DEVI;

	// write out
	for (int i = 0; i < TOTAL; i++)
	{
		float inVal = *(input_data_ptr + i);

		*(output_data_ptr + i) = (inVal - ave) / sqrt(DEVI + 0.000001);
	}
	*/
	////////////////////////////////////////////////////////////
	for (int z = 0; z < output_dim[2]; z++)
	{
		long SKIP = z * (input_dim[0] * input_dim[1]);

		// calc average
		float ave = 0.0;
		float S = 0.0;

		int TOTAL = input_dim[0] * input_dim[1];
		// sum up input data
		for (int i = 0; i < TOTAL; i++)
		{
			S += *(input_data_ptr + i + SKIP);
		}

		// average ************
		ave = S / (float)TOTAL;
		//*********************


		//calc deviation
		float DEVI = 0.0;

		for (int i = 0; i < TOTAL; i++)
		{
			float TEMP = (*(input_data_ptr + i + SKIP) - ave);
			DEVI += (TEMP*TEMP);
		}

		// deviation ************
		DEVI /= (float)TOTAL;
		// **********************


		// hold deviation for back p
		*(FILTER_hold_DEVI_each_ptr + z) = DEVI;


		// write out
		float SCALE = *(FILTER_batchN_scale_ptr + z);
		float BIAS = *(FILTER_batchN_bias_ptr + z);

		for (int i = 0; i < TOTAL; i++)
		{
			float inVal = *(input_data_ptr + i + SKIP);
			float batchVal = (inVal - ave) / sqrt(DEVI + 0.0000001);

			// hold mid value for back propagation
			*(FILTER_batchN_hold_midVal + SKIP + i) = batchVal;

			*(output_data_ptr + SKIP + i) = (SCALE * batchVal) + BIAS;
		}
	
	}
}



void LayerClass::FILTER_softmax(float* inPtr)
{
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	// find max
	float M = -100.0;
	for (int i = 0; i < TOTAL; i++)
	{
		if (M < *(input_data_ptr + i))
		{
			M = *(input_data_ptr + i);
		}
	}


	// sumExp
	float sumExp = 0.0;
	for (int i = 0; i < TOTAL; i++)
	{
		float expVal = exp(*(input_data_ptr + i) - M);
		sumExp += expVal;

		// hold each exp for backP
		*(FILTER_hold_exp + i) = expVal;
	}

	// hold expSum for backP
	*FILTER_hold_expSum = sumExp;

	// write out
	for (int i = 0; i < TOTAL; i++)
	{
		float inVal = *(input_data_ptr + i);

		*(output_data_ptr + i) = exp(inVal - M) / sumExp;
	}
}



void LayerClass::FILTER_ReLU(float* inPtr)
{
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	for (int i = 0; i < TOTAL; i++)
	{
		float inVal = *(input_data_ptr + i);

		if (inVal >= 0.0)
		{
			*(output_data_ptr + i) = inVal;
		}
		else
		{
			//*(output_data_ptr + i) = 0.0;
			// Leaky ReLU**************
			*(output_data_ptr + i) = inVal * 0.05;
		}
	}
}



////////////////////////////////////////////////////
/////////// back propagation ///////////////////////
////////////////////////////////////////////////////



void LayerClass::back_propagation_FILTER(float* backInPtr)
{
	// set back-in ptr
	back_in_ptr = backInPtr;

	switch (FILTER_mode)
	{
	case 0:
		this->FILTER_back_batch_norm(backInPtr);
		break;
	case 1:
		this->FILTER_back_softmax(backInPtr);
		break;
	case 2:
		this->FILTER_back_ReLU(backInPtr);
		break;
	case 3:
		break;
	}
}





void LayerClass::FILTER_back_batch_norm(float* backInPtr)
{
	// A[0] = (back[0] / devi) - ( sqrt(devi)/devi^2 * (1.0/m) * prev[0] * sum(prev[k]*back[k]));
	// R[0] = A[0] + (1/m)sum(A[k])
	/*
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	// alloc A
	float* A = (float*)malloc(sizeof(float) * TOTAL);

	// calc SUM( back-in[k] * prevOut[k] )
	float sumBack = 0.0;
	for (int i = 0; i < TOTAL; i++)
	{
		float backVal = *(back_in_ptr + i);
		float prevOut = *(output_data_ptr + i);

		sumBack += backVal * prevOut;
	}


	// calc each A
	float DEVI = *FILTER_hold_DEVI_ptr;

	// constant
	float C = sqrt(DEVI) / (DEVI*DEVI);
	float m = 1.0 / (float)TOTAL;

	for (int i = 0; i < TOTAL; i++)
	{
		float backVal = *(back_in_ptr + i);
		float prevOut = *(output_data_ptr + i);
		float B = (backVal / DEVI);
		float D = prevOut * sumBack;

		*(A + i) = B - (C * m * D);
	}


	// sum A
	float sumA = 0.0;
	for (int i = 0; i < TOTAL; i++)
	{
		sumA += *(A + i);
	}

	sumA /= (float)TOTAL;

	// write out
	for (int i = 0; i < TOTAL; i++)
	{
		*(back_out_ptr + i) = *(A + i) + sumA;
	}

	free(A);
	*/
	//////////////////////////////////////////////////
	for (int z = 0; z < output_dim[2]; z++)
	{
		int SKIP = z * (input_dim[0] * input_dim[1]);
		int TOTAL = input_dim[0] * input_dim[1];

		// bias delta is back-in val itself
		for (int i = 0; i < TOTAL; i++)
		{
			float backVal = *(back_in_ptr + SKIP + i);
			*(FILTER_batchN_bias_delta_sum + z) += backVal;
		}

		//:::::::::::::::::::::::::::::::::::::::::::::::::

		// bias scale = midVal * backIn
		for (int i = 0; i < TOTAL; i++)
		{
			float backVal = *(back_in_ptr + SKIP + i);
			float midVal = *(FILTER_batchN_hold_midVal + SKIP + i);

			*(FILTER_batchN_scale_delta_sum + z) += backVal * midVal;
		}

		//:::::::::::::::::::::::::::::::::::::::::::::::::::


		// alloc A
		float* A = (float*)malloc(sizeof(float) * TOTAL);

		// calc SUM ( back-in[k] * prevOut[k] )
		float sumBack = 0.0;
		float SCALE = *(FILTER_batchN_scale_ptr + z);

		for (int i = 0; i < TOTAL; i++)
		{
			float backVal = (*(back_in_ptr + i + SKIP)) * SCALE;
			float prevOut = *(output_data_ptr + i + SKIP);

			sumBack += backVal * prevOut;
		}

		// calc each A
		float DEVI = *(FILTER_hold_DEVI_each_ptr + z);

		//constant
		float C = sqrt(DEVI) / (DEVI*DEVI);
		float m = 1.0 / (float)TOTAL;
		for (int i = 0; i < TOTAL; i++)
		{
			float backVal = *(back_in_ptr + i + SKIP);
			float prevOut = *(output_data_ptr + i + SKIP);
			float B = (backVal / DEVI);
			float D = prevOut * sumBack;

			*(A + i) = B - (C*m*D);
		}

		// sum A
		float sumA = 0.0;
		for (int i = 0; i < TOTAL; i++)
		{
			sumA += *(A + i);
		}

		sumA /= (float)TOTAL;

		// write out
		for (int i = 0; i < TOTAL; i++)
		{
			*(back_out_ptr + i + SKIP) = *(A + i) + sumA;
		}

		// free
		free(A);

	}// for out z

}




void LayerClass::FILTER_back_softmax(float* backInPtr)
{
	// back = prevExp[0]*back[0] - sum(prevExp[k]*back[k]);
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	float sumBack = 0.0;

	for (int i = 0; i < TOTAL; i++)
	{
		float prevExp = *(FILTER_hold_exp + i);
		float backVal = *(back_in_ptr + i);

		sumBack += prevExp * backVal;
	}


	// -1.0 / S^2
	float S = *FILTER_hold_expSum;
	sumBack /= (S*S);

	// calc each
	for (int i = 0; i < TOTAL; i++)
	{
		float prevExp = *(FILTER_hold_exp + i);
		float backVal = *(back_in_ptr + i);

		*(back_out_ptr + i) = ((backVal / S) - sumBack) * prevExp;
	}
}



void LayerClass::FILTER_back_ReLU(float* backInPtr)
{
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	for (int i = 0; i < TOTAL; i++)
	{
		float coef = 1.0;
		float prevOut = *(output_data_ptr + i);

		if (prevOut <= 0.0)
		{
			//coef = 0.0;
			//**************************
			// Leaky ReLU
			coef = 0.05;
			//**************************
		}

		*(back_out_ptr + i) = (*(back_in_ptr + i)) * coef;
	}
}






// *>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>



void LayerClass::enqueue_forward_kernel_FILTER()
{
	long TOTAL = input_dim[0] * input_dim[1] * input_dim[2];
	int even_odd = 1;

	even_odd = this->find_divisor(TOTAL, 8);

	cl_int err;
	size_t off2D[2] = {0, 0};
	size_t work2D[2] = { TOTAL, num_GPU_img };
	size_t local2D[2] = { even_odd, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];


	switch (FILTER_mode)
	{
	case 0: // batch norm
		/*
		work2D[0] = 1;
		work2D[1] = num_GPU_img;
		local2D[0] = 1;
		local2D[1] = LOCAL_IMG;

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_batch_norm,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_batch_norm] fail...%d\n", err); }
		*/

		work2D[0] = output_dim[2];
		work2D[1] = num_GPU_img;
		local2D[0] = 1;
		local2D[1] = LOCAL_IMG;

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_batch_norm_each_z,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_batch_norm_each_z] fail...%d\n", err); }

		break;

	case 1: // softmax
		work2D[0] = 1;
		work2D[1] = num_GPU_img;
		local2D[0] = 1;
		local2D[1] = LOCAL_IMG;

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_softmax,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_softmax] fail...%d\n", err); }

		break;

	case 2: // ReLU

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_ReLU,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_FILTER_ReLU] fail.... %d\n", err); }

		break;
	case 3:
		break;
	}
}


void LayerClass::enqueue_back_kernel_FILTER()
{
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];
	int even_odd = 1;

	even_odd = this->find_divisor(TOTAL, 8);

	cl_int err;
	size_t off2D[2] = { 0, 0 };
	size_t work2D[2] = { TOTAL, num_GPU_img };
	size_t local2D[2] = { even_odd, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];


	switch (FILTER_mode)
	{
	case 0:
		/*
		work2D[0] = 1;
		work2D[1] = num_GPU_img;
		local2D[0] = 1;
		local2D[1] = LOCAL_IMG;

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_back_batch_norm,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_back_batch_norm] fail.... %d\n", err); }
		*/
		work2D[0] = output_dim[2];
		work2D[1] = num_GPU_img;
		local2D[0] = 1;
		local2D[1] = LOCAL_IMG;

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_back_batch_norm_each_z,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_back_batch_norm_each_z] fail.... %d\n", err); }

		// sum bias
		work2D[0] = output_dim[2];
		work2D[1] = 1;
		local2D[0] = this->find_divisor(output_dim[2], 8);
		local2D[1] = 1;

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_back_batch_norm_sumDelta,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_back_batchN_sum_delta] fail.... %d\n", err); }

		break;

	case 1:
		work2D[0] = 1;
		work2D[1] = num_GPU_img;
		local2D[0] = 1;
		local2D[1] = LOCAL_IMG;

		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_back_softmax,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_back_softmax] fail.... %d\n", err); }
		break;

	case 2:
		err = clEnqueueNDRangeKernel(CMQ,
			krn_FILTER_back_ReLU,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_FILTER_back_ReLU] fail...%d\n", err); }
		break;

	case 3:
		break;
	
	}
}



void LayerClass::enqueue_read_back_FILTER()
{
	if (FILTER_mode == 0) // batch norm
	{
		// read back bias & scale delta sum
		cl_int err;
		cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

		err = clEnqueueReadBuffer(CMQ,
			mem_FILTER_batchN_bias_delta_sum,
			CL_FALSE, 0,
			sizeof(float)*output_dim[2],
			FILTER_batchN_bias_delta_sum,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("mem read-back[batch norm bias delta sum] fail...%d\n", err); }

		//////////////////////////////////////////////////////////////

		err = clEnqueueReadBuffer(CMQ,
			mem_FILTER_batchN_scale_delta_sum,
			CL_FALSE, 0,
			sizeof(float)*output_dim[2],
			FILTER_batchN_scale_delta_sum,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("mem read-back[batch norm scale delta sum] fail...%d\n", err); }

	}
}