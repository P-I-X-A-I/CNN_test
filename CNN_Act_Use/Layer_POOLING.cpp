#include "LayerClass.h"


void LayerClass::init_as_MAX_POOLING(int inX, int inY, int inZ, int poolMode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "POOLING");

	POOLING_mode = poolMode;// 0 max, 1 average

	input_dim[0] = inX;
	input_dim[1] = inY;
	input_dim[2] = inZ;
	output_dim[0] = inX / 2;
	output_dim[1] = inY / 2;
	output_dim[2] = inZ;

	printf( "init Layer as [%s]\n", layer_type_str);

	this->alloc_POOL_CPU_memory();
}


void LayerClass::alloc_POOL_CPU_memory()
{
	long dataSize;

	// input data ( set by prev )

	// output data
	dataSize = sizeof(float)*output_dim[0] * output_dim[1] * output_dim[2];
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	// back in( set by prev )

	// back out
	dataSize = sizeof(float) * input_dim[0] * input_dim[1] * input_dim[2];
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	// selected mask
	dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2];
	POOL_selected_mask_ptr = (float*)malloc(dataSize);
	memset(POOL_selected_mask_ptr, 0, dataSize);


	// set num saved parameter///////////////////////////////////////
	num_saved_param = 0;
	/////////////////////////////////////////////////////////////////
}




void LayerClass::setup_POOL_cl_mem()
{
	long dataSize;

	// input mem ( set by prev )

	// output mem
	dataSize = sizeof(float) * output_dim[0] * output_dim[1] * output_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_output_data, dataSize, CL_MEM_READ_WRITE);

	// back-in mem ( set by prev )

	// back-out mem
	dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_back_out, dataSize, CL_MEM_READ_WRITE);

	// selected mask
	dataSize = sizeof(float) * input_dim[0] * input_dim[1] * input_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_POOL_selected_mask, dataSize, CL_MEM_READ_WRITE);
}



void LayerClass::setup_POOL_cl_kernel()
{
	// forward process
	cl_obj->create_kernel_util(DEVICE_id, &krn_POOL_max_pooling, "max_pooling");
	cl_obj->set_kernel_arg_mem(&krn_POOL_max_pooling, 0, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_POOL_max_pooling, 1, &mem_POOL_selected_mask);
	cl_obj->set_kernel_arg_mem(&krn_POOL_max_pooling, 2, &mem_output_data);
	cl_obj->set_kernel_arg_int_val(&krn_POOL_max_pooling, 3, input_dim[0]);
	cl_obj->set_kernel_arg_int_val(&krn_POOL_max_pooling, 4, input_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_POOL_max_pooling, 5, output_dim[2]); // inZ = outZ

	// backward process
	cl_obj->create_kernel_util(DEVICE_id, &krn_POOL_back_max_pooling, "back_max_pooling");
	cl_obj->set_kernel_arg_mem(&krn_POOL_back_max_pooling, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_POOL_back_max_pooling, 1, &mem_POOL_selected_mask);
	cl_obj->set_kernel_arg_mem(&krn_POOL_back_max_pooling, 2, &mem_back_out);
	cl_obj->set_kernel_arg_int_val(&krn_POOL_back_max_pooling, 3, input_dim[0]);
	cl_obj->set_kernel_arg_int_val(&krn_POOL_back_max_pooling, 4, input_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_POOL_back_max_pooling, 5, input_dim[2]); // inZ = outZ

}



void LayerClass::learn_CPU_POOLING(float* inPtr, float* ansPtr)
{
	input_data_ptr = inPtr;

	switch (POOLING_mode)
	{
	case 0:// max pooling
		
		  //+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*
		for (int z = 0; z < output_dim[2]; z++) // inZ = outZ
		{
			int inSkip = z * (input_dim[0] * input_dim[1]);
			int outSkip = z * (output_dim[0] * output_dim[1]);

			for (int OY = 0; OY < output_dim[1]; OY++)
			{
				// access to output y
				int oyID = OY * output_dim[0];

				// access to input y
				int iyID = (OY * 2) * input_dim[0];

				for (int OX = 0; OX < output_dim[0]; OX++)
				{
					// access to output x
					int oxID = OX;

					// access to input x
					int ixID = OX * 2;


					// input ( left-top ) access ID
					int iID = inSkip + iyID + ixID;

					// 4 access pointer to input
					float* p[4];
					p[0] = input_data_ptr + (iID);
					p[1] = p[0] + 1; // next ( right )
					p[2] = p[0] + input_dim[0]; // + x width
					p[3] = p[2] + 1; // next ( right )

					float* maskPtr[4];
					maskPtr[0] = POOL_selected_mask_ptr + iID;
					maskPtr[1] = maskPtr[0] + 1;
					maskPtr[2] = maskPtr[0] + input_dim[0];
					maskPtr[3] = maskPtr[2] + 1;



					float maxVal = -100.0;
					int maxIDX = 0;
					float tempVal = 0.0;

					// select max cell
					for (int m = 0; m < 4; m++)
					{
						tempVal = *(p[m]);

						if (maxVal < tempVal)
						{
							maxVal = tempVal;
							maxIDX = m;
						}

						// erase mask once
						*maskPtr[m] = 0.0;
					}


					// set max mask
					*maskPtr[maxIDX] = 1.0; // 0 - 3

					// output access ID
					int oID = outSkip + oyID + oxID;
					*(output_data_ptr + oID) = maxVal;
				}
			}
		}
		//+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*+*

		break;

	case 1: // may be added
		break;
	default:
		break;
	}
}


void LayerClass::back_propagation_POOLING(float* backInPtr)
{
	back_in_ptr = backInPtr;

	switch (POOLING_mode)
	{
	case 0: // max pooling
		for (int OZ = 0; OZ < output_dim[2]; OZ++)
		{
			// back-in access z
			int b_in_Skip = OZ * output_dim[0] * output_dim[1];
			// back-out access z
			int b_out_Skip = OZ * input_dim[0] * input_dim[1];
			

			for (int OY = 0; OY < output_dim[1]; OY++)
			{
				// back-in access y
				int iyID = OY * output_dim[0];
				// back-out access y
				int oyID = (OY * 2) * input_dim[0];


				for (int OX = 0; OX < output_dim[0]; OX++)
				{
					// back-in access x
					int biID = b_in_Skip + iyID + OX;
					float backVal = *(back_in_ptr + biID);


					// 4 pointers
					int boID = b_out_Skip + oyID + (OX * 2); // left-top corner pos
					float* b_out_ptr[4];
					float* mask_ptr[4];

					b_out_ptr[0] = back_out_ptr + boID;
					b_out_ptr[1] = b_out_ptr[0] + 1;
					b_out_ptr[2] = b_out_ptr[0] + input_dim[0];
					b_out_ptr[3] = b_out_ptr[2] + 1;

					mask_ptr[0] = POOL_selected_mask_ptr + boID;
					mask_ptr[1] = mask_ptr[0] + 1;
					mask_ptr[2] = mask_ptr[0] + input_dim[0];
					mask_ptr[3] = mask_ptr[2] + 1;

					// write back out
					*b_out_ptr[0] = backVal * (*mask_ptr[0]);
					*b_out_ptr[1] = backVal * (*mask_ptr[1]);
					*b_out_ptr[2] = backVal * (*mask_ptr[2]);
					*b_out_ptr[3] = backVal * (*mask_ptr[3]);
				}
			}
		}
		break;

	case 1: // may be implemented
		break;
	default:
		break;
	}
}



void LayerClass::enqueue_forward_kernel_POOLING()
{
	int evenOdd_ox = 1;
	int evenOdd_oy = 1;

	evenOdd_ox = this->find_divisor(output_dim[0], 4);
	evenOdd_oy = this->find_divisor(output_dim[1], 2);
	

	cl_int err;
	size_t off3D[3] = {0,0,0};
	size_t work3D[3] = {output_dim[0], output_dim[1], num_GPU_img};
	size_t local3D[3] = { evenOdd_ox, evenOdd_oy, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	err = clEnqueueNDRangeKernel(CMQ,
		krn_POOL_max_pooling,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("[krn_POOL_max_pooling] fail...%d\n", err);
	}
}


void LayerClass::enqueue_back_kernel_POOLING()
{
	int evenOdd_ox = 1;
	int evenOdd_oy = 1;

	evenOdd_ox = this->find_divisor(output_dim[0], 4);
	evenOdd_oy = this->find_divisor(output_dim[1], 2);

	cl_int err;
	size_t off3D[3] = { 0,0,0 };
	size_t work3D[3] = { output_dim[0], output_dim[1], num_GPU_img };
	size_t local3D[3] = { evenOdd_ox, evenOdd_oy, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	err = clEnqueueNDRangeKernel(CMQ,
		krn_POOL_back_max_pooling,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("[krn_POOL_back_max_pooling] fail...%d\n", err);
	}
}
