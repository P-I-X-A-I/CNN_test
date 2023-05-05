#include "LayerClass.h"


void LayerClass::init_as_CONVOLUTION(int inX, int inY, int inZ, int krnSize, int outZ, int weightMode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "CONVOLUTION");

	input_dim[0] = inX;
	input_dim[1] = inY;
	input_dim[2] = inZ;

	output_dim[0] = inX;
	output_dim[1] = inY;
	output_dim[2] = outZ;

	CONV_weight_mode = weightMode; // 0 Xavier, 1 He, 2 Const

	kernel_size = krnSize;

	const char* tempStr[3];
	tempStr[0] = "Xavier";
	tempStr[1] = "He";
	tempStr[2] = "Const";

	printf("init Layer as [%s (%s)]\n", layer_type_str, tempStr[CONV_weight_mode]);
		

	this->alloc_CONV_CPU_memory();
}


////////////////////////////////////////////////
///////// CPU MEMORY SETUP /////////////////////
////////////////////////////////////////////////


void LayerClass::alloc_CONV_CPU_memory()
{
	long dataSize;

	// input data ( set by prev )

	// output data
	dataSize = sizeof(float) * output_dim[0] * output_dim[1] * output_dim[2];
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	// back-in ( set by prev )

	// back-out
	dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2];
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	// conv bias
	dataSize = sizeof(float)*output_dim[2];
	CONV_bias_ptr = (float*)malloc(dataSize);
	memset(CONV_bias_ptr, 0, dataSize);

	// conv kernel
	dataSize = sizeof(float) * (kernel_size * kernel_size * input_dim[2]) * output_dim[2];
	CONV_kernel_ptr = (float*)malloc(dataSize);
	memset(CONV_kernel_ptr, 0, dataSize);


	// set num saved parameter///////////////////////////////////////
	num_saved_param = output_dim[2] + (kernel_size*kernel_size * input_dim[2]*output_dim[2]);
	/////////////////////////////////////////////////////////////////


	// conv kernel velocity
	dataSize = sizeof(float)*(kernel_size*kernel_size*input_dim[2]) * output_dim[2];
	CONV_kernel_velocity = (float*)malloc(dataSize);
	memset(CONV_kernel_velocity, 0, dataSize);

	// conv bias velocity
	dataSize = sizeof(float)*output_dim[2];
	CONV_bias_velocity = (float*)malloc(dataSize);
	memset(CONV_bias_velocity, 0, dataSize);

	// conv kernel delta sum
	dataSize = sizeof(float)*(kernel_size*kernel_size*input_dim[2]) * output_dim[2];
	CONV_kernel_delta_sum = (float*)malloc(dataSize);
	memset(CONV_kernel_delta_sum, 0, dataSize);

	// conv bias delta sum
	dataSize = sizeof(float)*output_dim[2];
	CONV_bias_delta_sum = (float*)malloc(dataSize);
	memset(CONV_bias_delta_sum, 0, dataSize);

	// conv kernel ADAG history
	dataSize = sizeof(float)*(kernel_size*kernel_size*input_dim[2]) * output_dim[2] * ADAG_HISTORY;
	CONV_kernel_ADAG_hisroty = (float*)malloc(dataSize);
	for (int i = 0; i < (kernel_size*kernel_size*input_dim[2]) * output_dim[2] * ADAG_HISTORY; i++)
	{
		*(CONV_kernel_ADAG_hisroty + i) = 1.0;
	}

	// conv bias ADAG history
	dataSize = sizeof(float)*output_dim[2] * ADAG_HISTORY;
	CONV_bias_ADAG_history = (float*)malloc(dataSize);
	for (int i = 0; i < output_dim[2] * ADAG_HISTORY; i++)
	{
		*(CONV_bias_ADAG_history + i) = 1.0;
	}



	// set initial weight
	this->generate_initial_kernel_bias();
}



void LayerClass::generate_initial_kernel_bias()
{
	// kernel deviation
	int numKernel = kernel_size * kernel_size * input_dim[2] * output_dim[2];
	int numDeviKernel = kernel_size * kernel_size * input_dim[2];
	//int numDeviKernel = input_dim[0]*input_dim[1]; // decided by num input
	float krn_DEVI;

	// bias deviation
	int numBias = output_dim[2];
	//int numDeviBias = kernel_size*kernel_size*input_dim[2]; // decided by num input
	//float bias_DEVI;

	switch (CONV_weight_mode)
	{
	case 0: // Xavier
		krn_DEVI = sqrt(1.0 / (float)numDeviKernel);
		//bias_DEVI = sqrt(1.0 / (float)numDeviBias);
		break;

	case 1: // He
		krn_DEVI = sqrt(2.0 / (float)numDeviKernel);
		//bias_DEVI = sqrt(2.0 / (float)numDeviBias);
		break;

	case 2: // constant
		krn_DEVI = 0.033;
		//bias_DEVI = 0.033;
		break;
	}

	// C++ random
	std::random_device rDevice; // normal random device
	std::mt19937 mersenne(rDevice()); // 32bit Mersenne Twister
	std::normal_distribution<> norm_dist_kernel(0.0, krn_DEVI); // 
	//std::normal_distribution<> norm_dist_bias(0.0, bias_DEVI);


	// write kernel
	for (int k = 0; k < numKernel; k++)
	{
		*(CONV_kernel_ptr + k) = norm_dist_kernel(mersenne);
	}

	// write bias
	for (int b = 0; b < numBias; b++)
	{
		//*(CONV_bias_ptr + b) = norm_dist_bias(mersenne);
		*(CONV_bias_ptr + b) = 0.0; // initial bias = 0.0
	}

}




void LayerClass::setup_CONV_cl_mem()
{
	long dataSize;
	long KERNEL_SIZE = kernel_size * kernel_size * input_dim[2] * output_dim[2];
	int HALF_K = (kernel_size - 1) / 2;
	// input mem ( set by prev )

	// input mem copy ( with padding )
	dataSize = sizeof(float)*(input_dim[0] + (HALF_K * 2))*(input_dim[1] + (HALF_K * 2))*input_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_pad_input, dataSize, CL_MEM_READ_WRITE); // init by 0


	// output mem
	dataSize = sizeof(float) * output_dim[0] * output_dim[1] * output_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_output_data, dataSize, CL_MEM_READ_WRITE);

	// back-in mem ( set by prev )
	

	// back-in copy ( with padding )
	dataSize = sizeof(float)*(output_dim[0] + (HALF_K * 2))*(output_dim[1] + (HALF_K * 2))*output_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_pad_back_in, dataSize, CL_MEM_READ_WRITE); // init by 0



	// back-out mem
	dataSize = sizeof(float) * input_dim[0] * input_dim[1] * input_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_back_out, dataSize, CL_MEM_READ_WRITE);

	//// kernel bias. ////////////////////

	// kernel ( NO IMAGE LAYER )
	dataSize = sizeof(float) * KERNEL_SIZE;
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_kernel, dataSize, CL_MEM_READ_ONLY);
	// copy kernel to cl_mem
	cl_obj->update_mem_contents_util(DEVICE_id, &mem_CONV_kernel, CONV_kernel_ptr, dataSize);

	// bias ( NO IMAGE LAYER )
	dataSize = sizeof(float) * output_dim[2];
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_bias, dataSize, CL_MEM_READ_ONLY);
	// copy bias to cl_mem
	cl_obj->update_mem_contents_util(DEVICE_id, &mem_CONV_bias, CONV_bias_ptr, dataSize);

	/////// kernel & bias, delta-each, delta-sum ///////////////

	// kernel delta each
	dataSize = sizeof(float) * KERNEL_SIZE * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_kernel_delta_each, dataSize, CL_MEM_READ_WRITE);
	
	// bias delta each
	dataSize = sizeof(float)*output_dim[2] * num_GPU_img;
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_bias_delta_each, dataSize, CL_MEM_READ_WRITE);

	// kernel delta sum ( NO IMAGE LAYER )
	dataSize = sizeof(float) * KERNEL_SIZE;
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_kernel_delta_sum, dataSize, CL_MEM_READ_WRITE);

	// bias delta sum ( NO IMAGE LAYER )
	dataSize = sizeof(float) * output_dim[2];
	cl_obj->create_mem_util(DEVICE_id, &mem_CONV_bias_delta_sum, dataSize, CL_MEM_READ_WRITE);
}




void LayerClass::setup_CONV_cl_kernel()
{
	int HALF_K = (kernel_size - 1) / 2;

	// input padding
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_copy_pad_input, "copy_pad_input");
	cl_obj->set_kernel_arg_mem(&krn_CONV_copy_pad_input, 0, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_CONV_copy_pad_input, 1, &mem_CONV_pad_input);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_copy_pad_input, 2, input_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_copy_pad_input, 3, (input_dim[1] + HALF_K + HALF_K));
	cl_obj->set_kernel_arg_int_val(&krn_CONV_copy_pad_input, 4, HALF_K);

	// convolution with padding
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_convolution_test, "convolution_test");
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution_test, 0, &mem_CONV_pad_input);
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution_test, 1, &mem_CONV_kernel);
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution_test, 2, &mem_CONV_bias);
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution_test, 3, &mem_output_data);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution_test, 4, kernel_size);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution_test, 5, HALF_K);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution_test, 6, output_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution_test, 7, input_dim[2]);




	// forward convolution
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_convolution, "convolution");
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution, 0, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution, 1, &mem_CONV_kernel);
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution, 2, &mem_CONV_bias);
	cl_obj->set_kernel_arg_mem(&krn_CONV_convolution, 3, &mem_output_data);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution, 4, kernel_size);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution, 5, HALF_K);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution, 6, input_dim[0]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution, 7, input_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution, 8, input_dim[2]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_convolution, 9, output_dim[2]);

	/////////////////////////////////////////////////////////

	// back bias each
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_back_bias_each, "conv_back_bias_each");
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_bias_each, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_bias_each, 1, &mem_CONV_bias_delta_each);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_bias_each, 2, output_dim[0]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_bias_each, 3, output_dim[1]);

	// back kernel each
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_back_kernel_each, "conv_back_kernel_each");
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_kernel_each, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_kernel_each, 1, &mem_input_data);
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_kernel_each, 2, &mem_CONV_kernel_delta_each);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_kernel_each, 3, input_dim[0]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_kernel_each, 4, input_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_kernel_each, 5, input_dim[2]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_kernel_each, 6, kernel_size);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_kernel_each, 7, HALF_K);

	// sum back bias
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_sum_back_bias, "conv_back_bias_sum");
	cl_obj->set_kernel_arg_mem(&krn_CONV_sum_back_bias, 0, &mem_CONV_bias_delta_each);
	cl_obj->set_kernel_arg_mem(&krn_CONV_sum_back_bias, 1, &mem_CONV_bias_delta_sum);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_sum_back_bias, 2, num_GPU_img);

	// sum back kernel
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_sum_back_kernel, "conv_back_kernel_weight_sum");
	cl_obj->set_kernel_arg_mem(&krn_CONV_sum_back_kernel, 0, &mem_CONV_kernel_delta_each);
	cl_obj->set_kernel_arg_mem(&krn_CONV_sum_back_kernel, 1, &mem_CONV_kernel_delta_sum);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_sum_back_kernel, 2, num_GPU_img);



	// copy pad back-in
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_copy_pad_back_in, "copy_pad_back_in");
	cl_obj->set_kernel_arg_mem(&krn_CONV_copy_pad_back_in, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_CONV_copy_pad_back_in, 1, &mem_CONV_pad_back_in);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_copy_pad_back_in, 2, output_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_copy_pad_back_in, 3, output_dim[1] + HALF_K + HALF_K);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_copy_pad_back_in, 4, HALF_K);


	// back out ( test )
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_back_out_test, "conv_back_out_test");
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_out_test, 0, &mem_CONV_pad_back_in);
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_out_test, 1, &mem_CONV_kernel);
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_out_test, 2, &mem_back_out);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out_test, 3, kernel_size);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out_test, 4, HALF_K);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out_test, 5, input_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out_test, 6, input_dim[2]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out_test, 7, output_dim[2]);


	// back out
	cl_obj->create_kernel_util(DEVICE_id, &krn_CONV_back_out, "conv_back_out");
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_out, 0, &mem_back_in);
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_out, 1, &mem_CONV_kernel);
	cl_obj->set_kernel_arg_mem(&krn_CONV_back_out, 2, &mem_back_out);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out, 3, kernel_size);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out, 4, HALF_K);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out, 5, input_dim[0] );
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out, 6, input_dim[1]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out, 7, input_dim[2]);
	cl_obj->set_kernel_arg_int_val(&krn_CONV_back_out, 8, output_dim[2]);

}



/////////////////////////////////////////////////////////////
////////////// CPU PROCESS //////////////////////////////////
/////////////////////////////////////////////////////////////


void LayerClass::learn_CPU_CONVOLUTION(float* inPtr, float* ansPtr)
{
	input_data_ptr = inPtr;

	int HALF_K = (kernel_size - 1) / 2;
	int K_SIZE = kernel_size * kernel_size * input_dim[2];

	for (int OZ = 0; OZ < output_dim[2]; OZ++)
	{
		// skip
		int krnSkip = K_SIZE * OZ;
		int outSkip = output_dim[0] * output_dim[1] * OZ;


		for (int OY = 0; OY < output_dim[1]; OY++)
		{
			// output access y
			int outPy = outSkip + (OY * output_dim[0]);


			for( int OX = 0 ; OX < output_dim[0] ; OX++ )
			{ 
				// putput position
				int outPos = outPy + OX;


				// kernel loop ::::::::::::::::::::::::::::::::::::::::
				float conVal = 0.0;

				for (int kz = 0; kz < input_dim[2]; kz++)
				{
					// kernel access z
					int KRNz = kz * kernel_size * kernel_size;

					// input access z
					int Iz = kz * input_dim[0] * input_dim[1];

					for (int ky = 0; ky < kernel_size; ky++)
					{
						// kernel access y
						int KRNy = ky * kernel_size;

						// input access y
						int Iy = OY + (ky - HALF_K);

						for (int kx = 0; kx < kernel_size; kx++)
						{
							// kernel access x
							int krnID = krnSkip + KRNz + KRNy + kx;
							float krnVal = *(CONV_kernel_ptr + krnID);



							// input access z
							int Ix = OX + (kx - HALF_K);
							int iID = Iz + (Iy * input_dim[0]) + Ix;
							float inVal = 0.0;

							// check range
							if (Ix >= 0 && Ix < input_dim[0])
							{
								if (Iy >= 0 && Iy < input_dim[1])
								{
									inVal = *(input_data_ptr + iID);
								}
							}

							// sumup
							conVal += inVal * krnVal;

						}
					}
				}
				// kernel loop ::::::::::::::::::::::::::::::::::::::::


				// add bias
				conVal += *(CONV_bias_ptr + OZ);

				// write to output pixel
				*(output_data_ptr + outPos) = conVal;
			} // ox
		} // oy
	} // oz

}




void LayerClass::back_propagation_CONVOLUTION(float* backInPtr)
{
	back_in_ptr = backInPtr;

	// back bias sum
	this->CONV_back_bias_sum();

	// back kernel sum
	this->CONV_back_kernel_sum();

	// back out
	this->CONV_back_out();
}



void LayerClass::CONV_back_bias_sum()
{
	for (int OZ = 0; OZ < output_dim[2]; OZ++)
	{
		float* backin_Head = back_in_ptr + (OZ * output_dim[0] * output_dim[1]);
		float backSum = 0.0;


		// sumup
		for (int OY = 0; OY < output_dim[1]; OY++)
		{
			for (int OX = 0; OX < output_dim[0]; OX++)
			{
				int accID = OY * output_dim[0] + OX;

				backSum += *(backin_Head + accID);
			}
		}

		// write out
		*(CONV_bias_delta_sum + OZ) += backSum;
	}
}



void LayerClass::CONV_back_kernel_sum()
{
	
	int KSIZE = kernel_size * kernel_size * input_dim[2];
	int HALF_K = (kernel_size - 1) / 2;


	for (int OZ = 0; OZ < output_dim[2]; OZ++)
	{
		// kernel skip
		int krnD_Head = OZ * KSIZE;
		int backIn_Head = OZ * output_dim[0] * output_dim[1];


		for (int OY = 0; OY < output_dim[1]; OY++)
		{
			for (int OX = 0; OX < output_dim[0]; OX++)
			{
				long biID = backIn_Head + (OY * output_dim[0]) + OX;
				float backVal = *(back_in_ptr + biID);



				// kernel loop :::::::::::::::::::::::::::::::::::::
				for (int KZ = 0; KZ < input_dim[2]; KZ++)
				{
					// access to kernel delta z
					int KDz = krnD_Head + (KZ * kernel_size * kernel_size);

					// access to input z
					int Izp = KZ * (input_dim[0] * input_dim[1]);


					for (int KY = 0; KY < kernel_size; KY++)
					{
						// access to kernel delta y
						int KDy = KY * kernel_size;

						// access to input y
						int Iy = OY + (KY - HALF_K);

						for (int KX = 0; KX < kernel_size; KX++)
						{
							// access to kernel delta
							int krn_del_ID = KDz + KDy + KX;

							// access to input x
							int Ix = OX + (KX - HALF_K);
							float prevIn = 0.0;

							// check range
							if (Ix >= 0 && Ix < input_dim[0])
							{
								if (Iy >= 0 && Iy < input_dim[1])
								{
									int iID = Izp + (Iy * input_dim[0]) + Ix;
									prevIn = *(input_data_ptr + iID);
								}
							}

							// sumup kernel delta
							*(CONV_kernel_delta_sum + krn_del_ID) += (backVal * prevIn);
						}
					}
				}
				// kernel loop :::::::::::::::::::::::::::::::::::::
			}
		}
	}
}



void LayerClass::CONV_back_out()
{
	// clear back-out
	long dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2];
	for (int i = 0; i < (input_dim[0]*input_dim[1]*input_dim[2]); i++)
	{
		*(back_out_ptr + i) = 0.0;
	}


	int HALF_K = (kernel_size - 1) / 2;


	for (int OZ = 0; OZ < output_dim[2]; OZ++)
	{
		int krn_Head = OZ * (kernel_size * kernel_size * input_dim[2]);
		int backin_Head = OZ * output_dim[0] * output_dim[1];
		

		// operate on every input xy pixel ( and z )
		for (int IY = 0; IY < input_dim[1]; IY++)
		{
			for (int IX = 0; IX < input_dim[0]; IX++)
			{
			
				// multiple kernel variable
				float backout_Sum[1024]; // maxmum inZ -> 1024

				for (int a = 0; a < 1024; a++)
				{
					backout_Sum[a] = 0.0; // initialize
				}


				// kernel loop :::::::::::::::::::::::::::
				for (int KZ = 0; KZ < input_dim[2]; KZ++)
				{
					// rev kernel access Z
					int RVKz = KZ * kernel_size * kernel_size;


					for (int KY = 0; KY < kernel_size; KY++)
					{
						// rev kernel access Y
						int RVKy = (kernel_size - 1) - KY;

						// back-in access Y
						int BIy = IY + (KY - HALF_K);


						for (int KX = 0; KX < kernel_size; KX++)
						{
							// rev kernel access X
							int RVKx = (kernel_size - 1) - KX;

							// rev kernel
							int revID = krn_Head + RVKz + (RVKy * kernel_size) + RVKx;
							float rev_krnVal = *(CONV_kernel_ptr + revID);

							/////////////////////////////////////////

							// back-in access X
							int BIx = IX + (KX - HALF_K);
							int biID = backin_Head + (BIy * output_dim[0]) + BIx;
							float backVal = 0.0;

							// check range
							if (BIx >= 0 && BIx < output_dim[0])
							{
								if (BIy >= 0 && BIy < output_dim[1])
								{
									backVal = *(back_in_ptr + biID);
								}
							}
							
							// sum up kernel * back-in
							backout_Sum[KZ] += (rev_krnVal * backVal);

						}
					}
				}
				// kernel loop :::::::::::::::::::::::::::


				// sumup to back-out
				for (int IZ = 0; IZ < input_dim[2]; IZ++)
				{
					int boID = (IZ * input_dim[0]*input_dim[1]) + (IY * input_dim[0]) + IX;

					*(back_out_ptr + boID) += backout_Sum[IZ];
				}


			}// inX
		}// inY

		
	}// OZ
}


//////////////////////////////////////////////////////////
///////////////// GPU PROCESS ////////////////////////////
//////////////////////////////////////////////////////////


void LayerClass::enqueue_forward_kernel_CONVOLUTION()
{
	int LOX = 1;
	int LOY = 1;
	int LOZ = 1;
	cl_int err;
	size_t off3D[3] = {0,0,0};
	size_t work3D[3] = { 1, 1, num_GPU_img };
	size_t local3D[3] = { 1, 1, LOCAL_IMG };

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	
	// copy into padding memory
	// inY * inZ * image parallel

	/*
	//////// COPY BUFFER into padding memory//////////////////////////
	int HALF_K = (kernel_size - 1) / 2;
	size_t src_origin[3] = { 0, 0, 0 };
	size_t dst_origin[3] = { HALF_K, HALF_K, 0 };
	size_t region[3] = { sizeof(float)*input_dim[0],
						sizeof(float)*input_dim[1],
						sizeof(float)*(input_dim[2] * num_GPU_img) };
	size_t src_row_pitch = sizeof(float)*input_dim[0];
	size_t src_slice_pitch = src_row_pitch * input_dim[1];
	size_t dst_row_pitch = sizeof(float)*(input_dim[0] + HALF_K + HALF_K);
	size_t dst_slice_pitch = dst_row_pitch * (input_dim[1] + HALF_K + HALF_K);

	clEnqueueCopyBufferRect(CMQ,
		mem_input_data,
		mem_CONV_pad_input,
		src_origin, dst_origin,
		region,
		src_row_pitch, src_slice_pitch,
		dst_row_pitch, dst_slice_pitch,
		0, NULL, NULL);
	/////////////////////////////////////////////////////////////////
	*/
	


	LOX = this->find_divisor(input_dim[0], 4);

	if( input_dim[2] != 1 ) // first input z may be 1
	{
		LOZ = this->find_divisor(input_dim[2], 2);
	}

	work3D[0] = input_dim[0];
	work3D[1] = input_dim[2];
	work3D[2] = num_GPU_img;
	local3D[0] = LOX;
	local3D[1] = LOZ;
	local3D[2] = LOCAL_IMG;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_copy_pad_input,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_copy_pad_input] fail...%d\n", err); }
	
	/////////////////////////////////////////////////////////////////////////

	
	// convolution
	// outX * outZ * image parallel

	LOX = 1;
	LOZ = 1;

	LOX = this->find_divisor(output_dim[0], 4);
	LOZ = this->find_divisor(output_dim[2], 2);


	work3D[0] = output_dim[0];
	work3D[1] = output_dim[2];
	work3D[2] = num_GPU_img;
	local3D[0] = LOX;
	local3D[1] = LOZ;
	local3D[2] = LOCAL_IMG;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_convolution_test,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_convolution] fail...%d\n", err); }
	
	
	/////////////////////////////////////////////////////////////////////////
	
	/*
	// convolution
	// outY * outZ * image parallel

	
	LOY = 1;
	LOZ = 1;

	LOY = this->find_divisor(output_dim[1], 2);
	LOZ = this->find_divisor(output_dim[2], 4);

	work3D[0] = output_dim[1];
	work3D[1] = output_dim[2];
	work3D[2] = num_GPU_img;
	local3D[0] = LOY;
	local3D[1] = LOZ;
	local3D[2] = LOCAL_IMG;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_convolution,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_convolution] fail...%d\n", err); }
	*/
}



void LayerClass::enqueue_back_kernel_CONVOLUTION()
{
	int evenOdd_in = 1;
	int evenOdd_out = 1;

	evenOdd_in = this->find_divisor(input_dim[2], 4);
	evenOdd_out = this->find_divisor(output_dim[2], 4);


	/////////////////////////////////////////////////

	cl_int err;
	size_t off2D[2] = {0, 0};
	size_t work2D[2] = {1,1};
	size_t local2D[2] = {1,1};

	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	// back bias each ( outZ * image parallel )
	work2D[0] = output_dim[2];
	work2D[1] = num_GPU_img;
	local2D[0] = evenOdd_out;
	local2D[1] = LOCAL_IMG;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_back_bias_each,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_back_bias_each] fail...%d\n", err); }

	///////////////////////////////////////////////////////////////

	// back kernel delta each ( outZ * image parallel )
	work2D[0] = output_dim[2];
	work2D[1] = num_GPU_img;
	local2D[0] = evenOdd_out;
	local2D[1] = LOCAL_IMG;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_back_kernel_each,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_back_kernel_each] fail...%d\n", err); }

	///////////////////////////////////////////////////////////////

	// sum up bias delta
	work2D[0] = output_dim[2];
	work2D[1] = 1;
	local2D[0] = evenOdd_out;
	local2D[1] = 1;

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_sum_back_bias,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_sum_back_bias] fail...%d\n", err); }

	///////////////////////////////////////////////////////////////

	// sum up kernel delta
	work2D[0] = kernel_size * kernel_size * input_dim[2] * output_dim[2];
	work2D[1] = 1;
	local2D[0] = evenOdd_out * evenOdd_in;
	local2D[1] = 1;

	// *+*+*+*+ CAUTION *********
	// kernel_delta_[each] is cleared in this kernel
	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_sum_back_kernel,
		2, off2D, work2D, local2D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_sum_back_kernel] fail...%d\n", err); }

	////////////////////////////////////////////////////////////////////
	
	// copy back-in ( with padding )
	
	// outX * outZ * image parallel
	size_t off3D[3] = { 0,0,0 };
	size_t work3D[3] = { input_dim[0],input_dim[1], num_GPU_img };
	size_t local3D[3] = { 1, 1, LOCAL_IMG };
	
	
	work3D[0] = input_dim[0];
	work3D[1] = input_dim[2];
	work3D[2] = num_GPU_img;

	local3D[0] = this->find_divisor(input_dim[0], 4);
	local3D[1] = this->find_divisor(input_dim[2], 2);


	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_copy_pad_back_in,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_pad_back_in] fail...%d\n", err); };

	////////////////////////////////////////////////////////////////////
	
	// back-out test ( inX * inZ * image parallel)
	work3D[0] = input_dim[0];
	work3D[1] = input_dim[2];
	work3D[2] = num_GPU_img;
	local3D[0] = 1;
	local3D[1] = 1;
	local3D[2] = LOCAL_IMG;

	local3D[0] = this->find_divisor(input_dim[0], 4);
	local3D[1] = this->find_divisor(input_dim[2], 2);

	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_back_out_test,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_back_out_test] fail...%d\n", err); };
	
	
	/*
	// back-out ( inX * inY * image parallel )
	
	work3D[0] = input_dim[0];
	work3D[1] = input_dim[1];
	work3D[2] = num_GPU_img;
	
	local3D[0] = this->find_divisor(input_dim[0], 4);
	local3D[1] = this->find_divisor(input_dim[1], 2);


	err = clEnqueueNDRangeKernel(CMQ,
		krn_CONV_back_out,
		3, off3D, work3D, local3D,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("[krn_CONV_back_out] fail...%d\n", err); };
	*/
}



void LayerClass::enqueue_read_back_CONVOLUTION()
{
	// read back kernel weight & bias delta sum
	cl_int err;
	cl_command_queue CMQ = cl_obj->cl_CMQ_obj[DEVICE_id];

	// read back bias delta sum
	err = clEnqueueReadBuffer(CMQ,
		mem_CONV_bias_delta_sum,
		CL_FALSE, 0,
		sizeof(float)*output_dim[2],
		CONV_bias_delta_sum,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("mem read-back [CONV_bias_delta_sum] fail...%d\n", err);}

	// read back kernel delta sum
	err = clEnqueueReadBuffer(CMQ,
		mem_CONV_kernel_delta_sum,
		CL_FALSE, 0,
		sizeof(float)*kernel_size*kernel_size*input_dim[2] * output_dim[2],
		CONV_kernel_delta_sum,
		0, NULL, NULL);

	if (err != CL_SUCCESS) { printf("mem read-back [CONV_kernel_delta_sum] fail...%d\n", err); }
}