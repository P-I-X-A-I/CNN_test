#include "LayerClass.h"


void LayerClass::init_as_CEMS(int inW, int mode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "CEMS");

	input_dim[0] = inW;
	input_dim[1] = 1;
	input_dim[2] = 1;
	output_dim[0] = 1;
	output_dim[1] = 1;
	output_dim[2] = 1;

	CEMS_mode = mode;

	const char* tempStr[3];
	tempStr[0] = "mean S";
	tempStr[1] = "cross E";
	tempStr[2] = "mean Abs S";

	printf("init Layer as [%s (%s)]\n", layer_type_str, tempStr[mode]);

	this->alloc_CEMS_CPU_memory();
}


void LayerClass::alloc_CEMS_CPU_memory()
{
	long dataSize;
	// input ptr ( set by prev )

	// output ptr ( final loss )
	dataSize = sizeof(float) * 1;
	output_data_ptr = (float*)malloc(dataSize);
	*output_data_ptr = 0.0;

	// back-int ptr ( NULL )

	// back-out ptr
	dataSize = sizeof(float) * input_dim[0];
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	// loss sum
	dataSize = sizeof(float) * 1;
	CEMS_loss_sum_ptr = (float*)malloc(dataSize);
	*CEMS_loss_sum_ptr = 0.0;
	
	// set num saved parameter///////////////////////////////////////
	num_saved_param = 0;
	/////////////////////////////////////////////////////////////////
}


/////////////////////////////////////////////////////////
/////////// OPEN CL SETUP ///////////////////////////////
/////////////////////////////////////////////////////////

void LayerClass::setup_CEMS_cl_mem()
{
	long dataSize;

	// input mem ( set by prev )

	// output mem
	dataSize = sizeof(float) * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_output_data, dataSize, CL_MEM_READ_WRITE);

	// back-in mem ( set by prev )

	// back-out mem
	dataSize = sizeof(float) * input_dim[0] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_back_out, dataSize, CL_MEM_READ_WRITE);

	// loss sum
	dataSize = sizeof(float) * 1;
	cl_obj->create_mem_util(DEVICE_id, &mem_CEMS_loss_sum, dataSize, CL_MEM_READ_WRITE);
}


void LayerClass::setup_CEMS_cl_kernel()
{
	// kernel
	cl_obj->create_kernel_util(DEVICE_id, &krn_CEMS_meanS, "meanS");
	cl_obj->set_kernel_arg_mem(&krn_CEMS_meanS, 0, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_CEMS_meanS, 1, &mem_answer);
	cl_obj->set_kernel_arg_int_val(&krn_CEMS_meanS, 2, input_dim[0]); // answer width
	cl_obj->set_kernel_arg_mem(&krn_CEMS_meanS, 3, &mem_output_data);
	cl_obj->set_kernel_arg_mem(&krn_CEMS_meanS, 4, &mem_back_out);

	// kernel
	cl_obj->create_kernel_util(DEVICE_id, &krn_CEMS_crossE, "crossE");
	cl_obj->set_kernel_arg_mem(&krn_CEMS_crossE, 0, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_CEMS_crossE, 1, &mem_answer);
	cl_obj->set_kernel_arg_int_val(&krn_CEMS_crossE, 2, input_dim[0]); // answer width
	cl_obj->set_kernel_arg_mem(&krn_CEMS_crossE, 3, &mem_output_data);
	cl_obj->set_kernel_arg_mem(&krn_CEMS_crossE, 4, &mem_back_out);

	// kernel
	cl_obj->create_kernel_util(DEVICE_id, &krn_CEMS_back_lossSum, "CEMS_lossSum");
	cl_obj->set_kernel_arg_mem(&krn_CEMS_back_lossSum, 0, &mem_output_data);
	cl_obj->set_kernel_arg_mem(&krn_CEMS_back_lossSum, 1, &mem_CEMS_loss_sum);
	cl_obj->set_kernel_arg_int_val(&krn_CEMS_back_lossSum, 2, num_GPU_img);
}


////////////////////////////////////////////////
/////////// CPU PROCESS ////////////////////////
////////////////////////////////////////////////


void LayerClass::learn_CPU_CEMS(float* inPtr, float* ansPtr)
{
	input_data_ptr = inPtr;
	answer_ptr = ansPtr;

	// clear loss value
	*output_data_ptr = 0.0;


	switch (CEMS_mode)
	{
	case 0: // meanS
		for (int i = 0; i < input_dim[0]; i++)
		{
			float inVal = *(input_data_ptr + i);
			float ansVal = *(answer_ptr + i);

			float meanS = (inVal - ansVal)*(inVal - ansVal);

			// sum up to output_data_ptr
			*output_data_ptr += meanS;

			// write back out
			*(back_out_ptr + i) = (inVal - ansVal);
		}

		*output_data_ptr *= 0.5;

		// sum loss value
		*CEMS_loss_sum_ptr += *output_data_ptr;

		break;

	case 1: // crossE
		for (int i = 0; i < input_dim[0]; i++)
		{
			float inVal = *(input_data_ptr + i); // invalue is softmaxed, always [+] value
			float ansVal = *(answer_ptr + i);

			// log ( 0.0 ) is infinity, so add small number
			float entropy = -(ansVal * log(inVal + 0.000001));

			// add to result loss
			*output_data_ptr += entropy;

			// write back out
			*(back_out_ptr + i) = (-ansVal) / (inVal + 0.000001);
		}

		// sum up loss value of all batch image
		*CEMS_loss_sum_ptr += *output_data_ptr;

		break;

	case 2:// mean abs S
		break;
	default:
		break;
	}
}


//////////////////////////////////////////////////
///////// GPU PROCESS ////////////////////////////
//////////////////////////////////////////////////


void LayerClass::enqueue_forward_kernel_CEMS()
{

	cl_int err;
	size_t off2D[2] = { 0,0 };
	size_t work2D[2] = { 1,num_GPU_img };
	size_t local2D[2] = {1, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	switch (CEMS_mode)
	{
	case 0: // mean S
		err = clEnqueueNDRangeKernel(CMQ,
			krn_CEMS_meanS,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_meanS] fail...%d\n", err); }
		break;

	case 1: // cross E
		err = clEnqueueNDRangeKernel(CMQ,
			krn_CEMS_crossE,
			2, off2D, work2D, local2D,
			0, NULL, NULL);

		if (err != CL_SUCCESS) { printf("[krn_crossE] fail...%d\n", err); }
		break;

	case 2: // mean ABS S
		break;

	default:
		break;
	}
}


void LayerClass::enqueue_back_kernel_CEMS()
{
	// sum up loss
	cl_int err;
	size_t off1D = 0;
	size_t work1D = 1;
	size_t local1D = 1;
	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CEMS_back_lossSum,
		1,
		&off1D, &work1D, &local1D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CEMS_lossSum] fail...%d\n", err); }

}


void LayerClass::enqueue_read_back_CEMS()
{
	cl_int err;

	err = clEnqueueReadBuffer(cl_obj->cl_CMQ_obj[DEVICE_id],
		mem_CEMS_loss_sum,
		CL_FALSE, 0,
		sizeof(float) * 1,
		CEMS_loss_sum_ptr,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("mem read-back [CEMS_loss_sum] fail...%d\n", err); }

}
