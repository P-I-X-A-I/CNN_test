#include "LayerClass.h"


void LayerClass::init_as_DROPOUT(int inX, int inY, int inZ, float RATE, int dropMode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "DROPOUT");

	input_dim[0] = inX;
	input_dim[1] = inY;
	input_dim[2] = inZ;
	output_dim[0] = inX;
	output_dim[1] = inY;
	output_dim[2] = inZ;

	DROPOUT_mode = dropMode; // 0 - training, 1-actual use (no countup, no gate)
	DROPOUT_rate = RATE;

	printf("init Layer as [%s]\n", layer_type_str);


	this->alloc_DROPOUT_CPU_memory();
}


void LayerClass::alloc_DROPOUT_CPU_memory()
{
	long dataSize;

	// in data ( set by prev )

	// out data
	dataSize = sizeof(float) * input_dim[0] * input_dim[1] * input_dim[2];
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	// back-in ( set by prev )

	// back-out
	dataSize = sizeof(float) * input_dim[0] * input_dim[1] * input_dim[2];
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	//////////////////////////////////////////////////////////////////

	// dropout gate
	dataSize = sizeof(float) * input_dim[0] * input_dim[1] * input_dim[2];
	DROP_gate_weight_ptr = (float*)malloc(dataSize);
	memset(DROP_gate_weight_ptr, 0, dataSize);

	// set num saved parameter///////////////////////////////////////
	num_saved_param = input_dim[0] * input_dim[1] * input_dim[2];
	/////////////////////////////////////////////////////////////////

	// dropout gate open count
	dataSize = sizeof(unsigned int) * input_dim[0] * input_dim[1] * input_dim[2];
	DROP_gate_open_count = (unsigned int*)malloc(dataSize);
	memset(DROP_gate_open_count, 0, dataSize);


	this->set_DROPOUT_initial_gate();
}





void LayerClass::set_DROPOUT_initial_gate()
{

	if (DROPOUT_mode == 0) // when training
	{
		// count up Total Count
		DROP_total_count++;

		// set gate weight
		long numGate = input_dim[0] * input_dim[1] * input_dim[2];

		for (int i = 0; i < numGate; i++)
		{

			float randVal = (float)(rand() % 10000) * 0.0001;

			if (randVal < DROPOUT_rate)
			{
				// set gate to 0.0
				*(DROP_gate_weight_ptr + i) = 0.0;
			}
			else
			{
				// set gate to 1.0
				*(DROP_gate_weight_ptr + i) = 1.0;
				// count up
				*(DROP_gate_open_count + i) += 1;
			}
		}
	}
	else // when act-use
	{
		// do nothing.
		// use gate weight from Save data.
	}
}



void LayerClass::setup_DROPOUT_cl_mem()
{
	long dataSize;
	long TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	// input mem ( set by prev )

	// output mem
	dataSize = sizeof(float) * TOTAL * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_output_data, dataSize, CL_MEM_READ_WRITE);

	// back-in mem ( set by prev )

	// back-outmem
	dataSize = sizeof(float) * TOTAL * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_back_out, dataSize, CL_MEM_READ_WRITE);

	//////////////////////////////////////////////////

	// gate weight ( NO IMAGE LAYER )
	dataSize = sizeof(float) * TOTAL;
	cl_obj->create_mem_util(DEVICE_id, &mem_DROP_gate_weight, dataSize, CL_MEM_READ_ONLY);
	cl_obj->update_mem_contents_util(DEVICE_id, &mem_DROP_gate_weight, DROP_gate_weight_ptr, dataSize);

}


void LayerClass::setup_DROPOUT_cl_kernel()
{
	// forward
	cl_obj->create_kernel_util(DEVICE_id, &krn_DROP_dropout, "dropout");
	cl_obj->set_kernel_arg_mem(&krn_DROP_dropout, 0, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_DROP_dropout, 1, &mem_DROP_gate_weight);
	cl_obj->set_kernel_arg_mem(&krn_DROP_dropout, 2, &mem_output_data);

	// back
	cl_obj->create_kernel_util(DEVICE_id, &krn_DROP_back_dropout, "back_dropout");
	cl_obj->set_kernel_arg_mem(&krn_DROP_back_dropout, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_DROP_back_dropout, 1, &mem_DROP_gate_weight);
	cl_obj->set_kernel_arg_mem(&krn_DROP_back_dropout, 2, &mem_back_out);
}



void LayerClass::learn_CPU_DROPOUT(float* inPtr, float* ansPtr)
{
	// set input pointer
	input_data_ptr = inPtr;


	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	for (int i = 0; i < TOTAL; i++)
	{
		// dropout gate
		float gateVal = *(DROP_gate_weight_ptr + i);
		float inVal = *(input_data_ptr + i);

		*(output_data_ptr + i) = inVal * gateVal;
	}
}



void LayerClass::back_propagation_DROPOUT(float* backInPtr)
{
	// set back-in pointer
	back_in_ptr = backInPtr;

	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	for (int i = 0; i < TOTAL; i++)
	{
		float gateVal = *(DROP_gate_weight_ptr + i);
		float backInVal = *(back_in_ptr + i);

		*(back_out_ptr + i) = backInVal * gateVal;
	}
}


//*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>


void LayerClass::enqueue_forward_kernel_DROPOUT()
{
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	int evenOdd = 1;
	evenOdd = this->find_divisor(TOTAL, 8);


	cl_int err;
	size_t off2D[2] = { 0, 0 };
	size_t work2D[2] = { TOTAL, num_GPU_img };
	size_t local2D[2] = { evenOdd, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	err = clEnqueueNDRangeKernel(CMQ,
		krn_DROP_dropout,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_dropout] fail...%d\n", err); }
}


void LayerClass::enqueue_back_kernel_DROPOUT()
{
	int TOTAL = input_dim[0] * input_dim[1] * input_dim[2];

	int evenOdd = 1;
	evenOdd = this->find_divisor(TOTAL, 8);

	cl_int err;
	size_t off2D[2] = { 0, 0 };
	size_t work2D[2] = { TOTAL, num_GPU_img };
	size_t local2D[2] = { evenOdd, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	err = clEnqueueNDRangeKernel(CMQ,
		krn_DROP_back_dropout,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_back_dropout] fail...%d\n", err); }
}