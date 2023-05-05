#include "LayerClass.h"


LayerClass::LayerClass()
{

}


LayerClass::~LayerClass()
{
	
}

////////////////////////////////////////////
//// SETUP METHOD //////////////////////////
////////////////////////////////////////////


void LayerClass::setup_cl_mem()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->setup_AFFINE_cl_mem();
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		this->setup_CEMS_cl_mem();
	}
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		this->setup_FILTER_cl_mem();
	}
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->setup_CONV_cl_mem();
	}
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
		this->setup_POOL_cl_mem();
	}
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->setup_DROPOUT_cl_mem();
	}
}


void LayerClass::setup_cl_kernel()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->setup_AFFINE_cl_kernel();
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		this->setup_CEMS_cl_kernel();
	}
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		this->setup_FILTER_cl_kernel();
	}
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->setup_CONV_cl_kernel();
	}
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
		this->setup_POOL_cl_kernel();
	}
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->setup_DROPOUT_cl_kernel();
	}
}



////////////////////////////////////////////
//// CPU METHOD //////////////////////////
////////////////////////////////////////////


void LayerClass::learn_CPU(float* inPtr, float* ansPtr)
{
	// ansPtr may not be used
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->learn_CPU_AFFINE(inPtr, ansPtr);
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		this->learn_CPU_CEMS(inPtr, ansPtr);
	}
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		this->learn_CPU_FILTER(inPtr, ansPtr);
	}
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->learn_CPU_CONVOLUTION(inPtr, ansPtr);
	}
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
		this->learn_CPU_POOLING(inPtr, ansPtr);
	}
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->learn_CPU_DROPOUT(inPtr, ansPtr);
	}
}


void LayerClass::back_propagation_CPU(float* backInPtr)
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->back_propagation_AFFINE(backInPtr);
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		// back-propagation is done in forward process
	}
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		this->back_propagation_FILTER(backInPtr);
	}
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->back_propagation_CONVOLUTION(backInPtr);
	}
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
		this->back_propagation_POOLING(backInPtr);
	}
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->back_propagation_DROPOUT(backInPtr);
	}
}



////////////////////////////////////////////
//// GPU METHOD //////////////////////////
////////////////////////////////////////////

void LayerClass::learn_CL()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->enqueue_forward_kernel_AFFINE();
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		this->enqueue_forward_kernel_CEMS();
	}
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		this->enqueue_forward_kernel_FILTER();
	}
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->enqueue_forward_kernel_CONVOLUTION();
	}
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
		this->enqueue_forward_kernel_POOLING();
	}
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->enqueue_forward_kernel_DROPOUT();
	}
}


void LayerClass::back_propagation_CL()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->enqueue_back_kernel_AFFINE();
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		this->enqueue_back_kernel_CEMS();
	}
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		this->enqueue_back_kernel_FILTER();
	}
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->enqueue_back_kernel_CONVOLUTION();
	}
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
		this->enqueue_back_kernel_POOLING();
	}
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->enqueue_back_kernel_DROPOUT();
	}
}


void LayerClass::readback_CL()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->enqueue_read_back_kernel_AFFINE();
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		this->enqueue_read_back_CEMS();
	}
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		this->enqueue_read_back_FILTER(); // if FILTER is [batch-norm]
	} 
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->enqueue_read_back_CONVOLUTION();
	}
	else if (strcmp("POOLING", layer_type_str) == 0)
	{} // do nothing
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{} // do nothing

}


////////////////////////////////////////////
//// UPDATE METHOD //////////////////////////
////////////////////////////////////////////


void LayerClass::copy_weight_bias_from(LayerClass* srcLayer)
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// copy weight & bias
		float* srcWeight = srcLayer->AFFINE_weight_ptr;
		float* srcBias = srcLayer->AFFINE_bias_ptr;
		long dataSize_wei = sizeof(float)*input_dim[0] * output_dim[0];
		long dataSize_bias = sizeof(float)*output_dim[0];

		// copy weight
		memcpy(AFFINE_weight_ptr, srcWeight, dataSize_wei);
		// copy bias
		memcpy(AFFINE_bias_ptr, srcBias, dataSize_bias);
	}
	////////////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{} // do nothing 
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// if batch norm
		if (FILTER_mode == 0)
		{
			// copy scale & bias
			float* srcBias = srcLayer->FILTER_batchN_bias_ptr;
			float* srcScale = srcLayer->FILTER_batchN_scale_ptr;
			long dataSize = sizeof(float)*output_dim[2]; // scale & bias size is same

			// copy bias
			memcpy(FILTER_batchN_bias_ptr, srcBias, dataSize);
			// copy scale
			memcpy(FILTER_batchN_scale_ptr, srcScale, dataSize);
		}
	}
	////////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// copy bias & kernel weight
		float* srcKrn = srcLayer->CONV_kernel_ptr;
		float* srcBias = srcLayer->CONV_bias_ptr;
		long dataSize_krn = sizeof(float)*(kernel_size * kernel_size * input_dim[2] * output_dim[2]);
		long dataSize_bias = sizeof(float) * output_dim[2];

		// copy kernel
		memcpy(CONV_kernel_ptr, srcKrn, dataSize_krn);
		// copy bias
		memcpy(CONV_bias_ptr, srcBias, dataSize_bias);
	}
	//////////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{} // do nothing
	//////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		// copy gate weight
		float* gatePtr = srcLayer->DROP_gate_weight_ptr;
		long dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2];

		// copy gate weight
		memcpy(DROP_gate_weight_ptr, gatePtr, dataSize);
	}
}








void LayerClass::add_weight_bias_lossSum(LayerClass* srcLayer)
{


	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// add bias & weight delta sum
		float* fromBias = srcLayer->AFFINE_bias_delta_sum;
		float* fromWeight = srcLayer->AFFINE_weight_delta_sum;

		// add bias delta sum
		for (int i = 0; i < output_dim[0]; i++)
		{
			*(AFFINE_bias_delta_sum + i) += *(fromBias + i);
		}
		// add weight delta sum
		for (int i = 0; i < input_dim[0] * output_dim[0]; i++)
		{
			*(AFFINE_weight_delta_sum + i) += *(fromWeight + i);
		}

	}
	/////////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		float* fromLossSum = srcLayer->CEMS_loss_sum_ptr;

		// add LossSum
		*CEMS_loss_sum_ptr += *fromLossSum;
	}
	////////////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// it batch norm
		if (FILTER_mode == 0)
		{
			// add scale & bias delta sum into one
			float* fromBias = srcLayer->FILTER_batchN_bias_delta_sum;
			float* fromScale = srcLayer->FILTER_batchN_scale_delta_sum;

			// add
			for (int i = 0; i < output_dim[2]; i++)
			{
				// bias
				*(FILTER_batchN_bias_delta_sum + i) += *(fromBias + i);
				// scale
				*(FILTER_batchN_scale_delta_sum + i) += *(fromScale + i);
			}
		}
	}
	////////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		float* srcKernel_delta_sum = srcLayer->CONV_kernel_delta_sum;
		float* srcBias_delta_sum = srcLayer->CONV_bias_delta_sum;

		// add kernel delta
		long KSIZE = kernel_size * kernel_size * input_dim[2] * output_dim[2];
		for (int i = 0; i < KSIZE; i++)
		{
			*(CONV_kernel_delta_sum + i) += *(srcKernel_delta_sum + i);
		}

		// add bias delta
		for (int i = 0; i < output_dim[2]; i++)
		{
			*(CONV_bias_delta_sum + i) += *(srcBias_delta_sum + i);
		}
	}
	//////////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{} // do nothing
   //////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{} // do nothing
}







void LayerClass::average_weight_bias_lossSum(float coef)
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// average bias delta
		for (int i = 0; i < output_dim[0]; i++)
		{
			*(AFFINE_bias_delta_sum + i) *= coef; // divided by NUM_BATCH_IMAGE
		}
		// average weight delta
		for (int i = 0; i < input_dim[0] * output_dim[0]; i++)
		{
			*(AFFINE_weight_delta_sum + i) *= coef; // divided by NUM_BATCH_IMAGE
		}
	}
	////////////////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		(*CEMS_loss_sum_ptr) *= coef; // divided by NUM_BATCH_IMAGE
	}
	////////////////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// if batch norm
		if (FILTER_mode == 0)
		{
			for (int i = 0; i < output_dim[2]; i++)
			{
				*(FILTER_batchN_bias_delta_sum + i) *= coef; // divided by NUM_BATCH_IMAGE
				*(FILTER_batchN_scale_delta_sum + i) *= coef;
			}
		}
	}
	////////////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// average kernel delta sum
		int KSIZE = kernel_size * kernel_size * input_dim[2] * output_dim[2];
		for (int i = 0; i < KSIZE; i++)
		{
			*(CONV_kernel_delta_sum + i) *= coef; // divided by NUM_BATCH_IMAGE
		}

		// average bias delta sum
		for (int i = 0; i < output_dim[2]; i++)
		{
			*(CONV_bias_delta_sum + i) *= coef; // divided by NUM_BATCh_IMAGE
		}
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{} // do nothing
   //////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{} // do nothing
}




void LayerClass::change_dropout_gate()
{
	// only for dropout layer
	if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->set_DROPOUT_initial_gate();
	}
}





void LayerClass::weight_decay(float decayCoef)
{
	// weight decay done on GPU0 chain after delta (sum->average) calculation 
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// weight decay is adapted only on [Weight delta] not [bias delta]
		for (int i = 0; i < input_dim[0] * output_dim[0]; i++)
		{
			*(AFFINE_weight_delta_sum + i) += (*(AFFINE_weight_ptr + i))*decayCoef;
		}


		// calc L2 norm
		float len = 0.0;
		for (int i = 0; i < (input_dim[0] * output_dim[0]); i++)
		{
			float weightVal = *(AFFINE_weight_ptr + i);
			len += (weightVal * weightVal);
		}

		// set common weight decay val
		weight_decay_val = sqrt(len) * 0.5 * decayCoef;

	}
	//////////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
	}
	///////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		//if batch norm
		if (FILTER_mode == 0)
		{
			// weight decay is adapted only on [scale], not [bias]
			for (int i = 0; i < output_dim[2]; i++)
			{
				*(FILTER_batchN_scale_delta_sum + i) += (*(FILTER_batchN_scale_ptr + i))*decayCoef;
			}


			// calc L2 norm
			float len = 0.0;
			for (int i = 0; i < output_dim[2]; i++)
			{
				float scaleVal = *(FILTER_batchN_scale_ptr + i);
				len += scaleVal * scaleVal;
			}

			// set common weight decay val
			weight_decay_val = sqrt(len) * 0.5 * decayCoef;
		}
	}
	///////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// weight decay done on GPU0 chain after delta (sum->average) calculation 
		int KSIZE = kernel_size * kernel_size * input_dim[2];

		// weight decay effect only on kernel-weight, not bias.
		int acc = 0;
		for (int z = 0; z < output_dim[2]; z++)
		{
			for (int k = 0; k < KSIZE; k++)
			{
				*(CONV_kernel_delta_sum + acc) += (*(CONV_kernel_ptr + acc))*decayCoef;
				acc++;
			}
		}


		// calc weight decay

		float decaySum = 0.0;
		acc = 0;
		for (int z = 0; z < output_dim[2]; z++)
		{
			for (int k = 0; k < KSIZE; k++)
			{
				float krnVal = *(CONV_kernel_ptr + acc);
				acc++;

				decaySum += (krnVal * krnVal);
			}
		}

		// set weight decay val
		weight_decay_val = sqrt(decaySum) * decayCoef * 0.5;
	}
	//////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
  //////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
	} // do nothing
}




void LayerClass::update_cl_mem_weight()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		long dataSize;

		// update weight, bias cl_mem contents
		dataSize = sizeof(float)*output_dim[0];
		cl_obj->update_mem_contents_util(DEVICE_id, &mem_AFFINE_bias, AFFINE_bias_ptr, dataSize);

		dataSize = sizeof(float)*input_dim[0] * output_dim[0];
		cl_obj->update_mem_contents_util(DEVICE_id, &mem_AFFINE_weight, AFFINE_weight_ptr, dataSize);

	}
	////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{} // do nothing.....::::::::
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// update bias & scale val
		if (FILTER_mode == 0) // batch norm
		{
			long dataSize;

			// bias
			dataSize = sizeof(float) * output_dim[2];
			cl_obj->update_mem_contents_util(DEVICE_id, &mem_FILTER_batchN_bias, FILTER_batchN_bias_ptr, dataSize);

			dataSize = sizeof(float)*output_dim[2];
			cl_obj->update_mem_contents_util(DEVICE_id, &mem_FILTER_batchN_scale, FILTER_batchN_scale_ptr, dataSize);
		}
	}
	///////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		long dataSize;

		// update bias & kernel-weight
		dataSize = sizeof(float)*output_dim[2];
		cl_obj->update_mem_contents_util(DEVICE_id, &mem_CONV_bias, CONV_bias_ptr, dataSize);

		dataSize = sizeof(float)*kernel_size*kernel_size * input_dim[2] * output_dim[2];
		cl_obj->update_mem_contents_util(DEVICE_id, &mem_CONV_kernel, CONV_kernel_ptr, dataSize);
	}
	///////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
  //////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		// update gate weight
		long dataSize = sizeof(float)*input_dim[0] * input_dim[1] * input_dim[2];
		cl_obj->update_mem_contents_util(DEVICE_id, &mem_DROP_gate_weight, DROP_gate_weight_ptr, dataSize);
	}
}



void LayerClass::clear_sum()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// clear weight & bias delta sum
		for (int i = 0; i < output_dim[0]; i++)
		{
			*(AFFINE_bias_delta_sum + i) = 0.0;
		}
		for (int i = 0; i < input_dim[0] * output_dim[1]; i++)
		{
			*(AFFINE_weight_delta_sum + i) = 0.0;
		}
	}
	////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		// clear loss sum
		*CEMS_loss_sum_ptr = 0.0;
	}
	////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// if batch norm
		if (FILTER_mode == 0)
		{
			// clear bias & scale delta sum
			for (int i = 0; i < output_dim[2]; i++)
			{
				*(FILTER_batchN_bias_delta_sum + i) = 0.0;
				*(FILTER_batchN_scale_delta_sum + i) = 0.0;
			}
		}
	}
	////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// clear kernel & bias delta sum
		int dataCount = kernel_size * kernel_size * input_dim[2] * output_dim[2];
		for (int i = 0; i < dataCount; i++)
		{
			*(CONV_kernel_delta_sum + i) = 0.0;
		}

		// clear bias delta sum
		for (int i = 0; i < output_dim[2]; i++)
		{
			*(CONV_bias_delta_sum + i) = 0.0;
		}
	}
	////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
  //////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
	} // do nothing
}





////////////////////////////////////////////
//// weight update METHOD //////////////////////////
////////////////////////////////////////////

void LayerClass::update_by_SDG(float coef)
{
	////////////////////////////////////////////////////////
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// update bias
		for (int i = 0; i < output_dim[0]; i++)
		{
			float biasDelta = *(AFFINE_bias_delta_sum + i);

			// limit delta
			biasDelta = this->limit_delta(biasDelta * coef);

			// update bias
			*(AFFINE_bias_ptr + i) -= biasDelta;
		}

		// update weight
		for (int i = 0; i < (input_dim[0] * output_dim[0]); i++)
		{
			float weightDelta = *(AFFINE_weight_delta_sum + i);

			// limit delta
			weightDelta = this->limit_delta(weightDelta * coef);

			// update weight
			*(AFFINE_weight_ptr + i) -= weightDelta;
		}
	}
	///////////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{} // do nothing .........
	//////////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0) 
	{
		// if batch norm
		if (FILTER_mode == 0)
		{
			//update bias & scale
			for (int i = 0; i < output_dim[2]; i++)
			{
				float biasDelta = *(FILTER_batchN_bias_delta_sum + i);

				// limit delta
				biasDelta = this->limit_delta(biasDelta * coef);

				// update bias
				*(FILTER_batchN_bias_ptr + i) -= biasDelta;

				////////////////////////////////////////////////////

				float scaleDelta = *(FILTER_batchN_scale_delta_sum + i);

				// limit delta
				scaleDelta = this->limit_delta(scaleDelta * coef);

				// update scale
				*(FILTER_batchN_scale_ptr + i) -= scaleDelta;

			}
		}
	}
	///////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// updata bias
		for (int i = 0; i < output_dim[2]; i++)
		{
			float bias_delta = *(CONV_bias_delta_sum + i); // averaged

			// limit delta
			bias_delta = this->limit_delta(bias_delta * coef);

			*(CONV_bias_ptr + i) -= bias_delta;
		}

		// update kernel weight
		int KSIZE = kernel_size * kernel_size * input_dim[2] * output_dim[2];
		for (int i = 0; i < KSIZE; i++)
		{
			float kernel_delta = *(CONV_kernel_delta_sum + i); // averaged

			// limit delta
			kernel_delta = this->limit_delta(kernel_delta * coef);

			*(CONV_kernel_ptr + i) -= kernel_delta;
		}
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
  //////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
	} // do nothing
}



void LayerClass::update_by_Momemtom(float coef, float brake)
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// update bias
		for (int i = 0; i < output_dim[0]; i++)
		{
			// brake
			*(AFFINE_bias_velocity + i) *= brake;

			// add F to velocity
			float biasDelta = *(AFFINE_bias_delta_sum + i);

			// limit delta
			biasDelta = this->limit_delta(biasDelta  * coef);

			*(AFFINE_bias_velocity + i) -= biasDelta;

			// modify weight by velocity
			*(AFFINE_bias_ptr + i) += *(AFFINE_bias_velocity + i);
		}

		// update weight
		for (int i = 0; i < input_dim[0] * output_dim[0]; i++)
		{
			// brake velocity
			*(AFFINE_weight_velocity + i) *= brake;

			// add F to velocity
			float weightDelta = *(AFFINE_weight_delta_sum + i);

			// limit delta
			weightDelta = this->limit_delta(weightDelta * coef);

			*(AFFINE_weight_velocity + i) -= weightDelta;

			// modify weight by velocity
			*(AFFINE_weight_ptr + i) += *(AFFINE_weight_velocity + i);
		}
	}
	//////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{} // do nothing......
	///////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// if batch norm
		if (FILTER_mode == 0)
		{
			for (int i = 0; i < output_dim[2]; i++) // update bias & scale by velocity
			{
				// brake bias
				*(FILTER_batchN_bias_velocity + i) *= brake;

				// add F to velocity
				float biasDelta = *(FILTER_batchN_bias_delta_sum + i);
				biasDelta = this->limit_delta(biasDelta * coef);

				*(FILTER_batchN_bias_velocity + i) -= biasDelta;

				// modify bias by velocity
				*(FILTER_batchN_bias_ptr + i) += *(FILTER_batchN_bias_velocity + i);

				///////////////////////////////////////////////////////////////////////

				// brake scale
				*(FILTER_batchN_scale_velocity + i) *= brake;

				// add F to velocity
				float scaleDelta = *(FILTER_batchN_scale_delta_sum + i);
				scaleDelta = this->limit_delta(scaleDelta * coef);

				*(FILTER_batchN_scale_velocity + i) -= scaleDelta;

				// modify scale by velocity
				*(FILTER_batchN_scale_ptr + i) += *(FILTER_batchN_scale_velocity + i);
			}
		}
	}
	/////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// update bias by velocity
		for (int i = 0; i < output_dim[2]; i++)
		{
			// brake bias velocity
			*(CONV_bias_velocity + i) *= brake;

			// add F to velocity
			float bias_delta = *(CONV_bias_delta_sum + i);

			// limit delta
			bias_delta = this->limit_delta(bias_delta * coef);

			//
			*(CONV_bias_velocity + i) -= bias_delta;

			// modify bias by velocity
			*(CONV_bias_ptr + i) += *(CONV_bias_velocity + i);
		}


		// update kernel-weight by velocity
		int kCount = kernel_size * kernel_size * input_dim[2] * output_dim[2];
		for (int i = 0; i < kCount; i++)
		{
			// brake velocity
			*(CONV_kernel_velocity + i) *= brake;

			// add F to velocity
			float kernel_delta = *(CONV_kernel_delta_sum + i);

			// limit delta
			kernel_delta = this->limit_delta(kernel_delta * coef);

			*(CONV_kernel_velocity + i) -= kernel_delta;

			// modify kernel weight by velocity
			*(CONV_kernel_ptr + i) += *(CONV_kernel_velocity + i);
		}
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
    //////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
	} // do nothing


}



void LayerClass::update_by_AdaGrad(float coef)
{
	float INI_Coef = 0.001;

	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		for (int i = 0; i < output_dim[0]; i++)
		{
			// shift ADAG History
			this->shift_ADAG_History(AFFINE_bias_ADAG_history + (i * ADAG_HISTORY), ADAG_HISTORY);

			// set ADAG History to History[0]
			float biasDelta = *(AFFINE_bias_delta_sum + i);
			*(AFFINE_bias_ADAG_history + (i * ADAG_HISTORY)) = biasDelta * biasDelta;
		}

		for (int i = 0; i < input_dim[0] * output_dim[1]; i++)
		{
			// shift ADAG History
			this->shift_ADAG_History(AFFINE_weight_ADAG_history + (i*ADAG_HISTORY), ADAG_HISTORY);

			// set ADAG History to History[0]
			float weightDelta = *(AFFINE_weight_delta_sum + i);
			*(AFFINE_weight_ADAG_history + (i*ADAG_HISTORY)) = weightDelta * weightDelta;
		}

		/////////////////////////////////////////////////////////////////////////


		// calculate bias
		for (int i = 0; i < output_dim[0]; i++)
		{
			float bias_Delta = *(AFFINE_bias_delta_sum + i);

			float ADAG_bias_SUM = 0.0;
			float ADAG_bias_coef = INI_Coef;
			for (int a = 0; a < ADAG_HISTORY; a++)
			{
				ADAG_bias_SUM += *(AFFINE_bias_ADAG_history + (i*ADAG_HISTORY) + a);
			}

			ADAG_bias_coef /= sqrt(ADAG_bias_SUM + 0.000001);

			//
			bias_Delta = bias_Delta * ADAG_bias_coef * coef;
			bias_Delta = this->limit_delta(bias_Delta);

			// update
			*(AFFINE_bias_ptr + i) -= bias_Delta;
		}


		// calculate weight
		for (int i = 0; i < input_dim[0] * output_dim[0]; i++)
		{
			float weight_Delta = *(AFFINE_weight_delta_sum + i);

			// sumup
			float ADAG_weight_SUM = 0.0;
			float ADAG_weight_coef = INI_Coef;
			for (int a = 0; a < ADAG_HISTORY; a++)
			{
				ADAG_weight_SUM += *(AFFINE_weight_ADAG_history + (i*ADAG_HISTORY) + a);
			}

			ADAG_weight_coef /= sqrt(ADAG_weight_SUM + 0.000001);

			//
			weight_Delta = weight_Delta * ADAG_weight_coef * coef;
			weight_Delta = this->limit_delta(weight_Delta);

			// update
			*(AFFINE_weight_ptr + i) -= weight_Delta;
		}

	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
	} // do nothing......
   ///////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// if batch norm
		if (FILTER_mode == 0)
		{
			// shift scale & bias ADAG history
			for (int i = 0; i < output_dim[2]; i++)
			{
				// shift bias ADAG history
				this->shift_ADAG_History(FILTER_batchN_bias_ADAG_history + (i*ADAG_HISTORY), ADAG_HISTORY);

				// set bias ADAG history[0];
				float biasDelta = *(FILTER_batchN_bias_delta_sum + i);
				*(FILTER_batchN_bias_ADAG_history + (i*ADAG_HISTORY)) = biasDelta * biasDelta;

				//////////////////////////////////////////////////////////////////////////////

				// shift scale ADAG history
				this->shift_ADAG_History(FILTER_batchN_scale_ADAG_history + (i*ADAG_HISTORY), ADAG_HISTORY);

				// set scale ADAG history[0]
				float scaleDelta = *(FILTER_batchN_scale_delta_sum + i);
				*(FILTER_batchN_scale_ADAG_history + (i*ADAG_HISTORY)) = scaleDelta * scaleDelta;
			}

			// calc
			for (int i = 0; i < output_dim[2]; i++)
			{
				// bias delta
				float biasDelta = *(FILTER_batchN_bias_delta_sum + i);

				// coef
				float ADAG_bias_SUM = 0.0;
				float ADAG_bias_Coef = INI_Coef;

				for (int a = 0; a < ADAG_HISTORY; a++)
				{
					ADAG_bias_SUM += *(FILTER_batchN_bias_ADAG_history + (i*ADAG_HISTORY) + a);
				}

				ADAG_bias_Coef /= sqrt(ADAG_bias_SUM + 0.000001);

				// calc delta
				biasDelta = biasDelta * ADAG_bias_Coef * coef;
				biasDelta = this->limit_delta(biasDelta);

				// update bias
				*(FILTER_batchN_bias_ptr + i) -= biasDelta;

				//////////////////////////////////////////////////////////////////////////////////////

				// scale delta
				float scaleDelta = *(FILTER_batchN_scale_delta_sum + i);

				// coef
				float ADAG_scale_SUM = 0.0;
				float ADAG_scale_Coef = INI_Coef;

				for (int a = 0; a < ADAG_HISTORY; a++)
				{
					ADAG_scale_SUM += *(FILTER_batchN_scale_ADAG_history + (i*ADAG_HISTORY) + a);
				}

				ADAG_scale_Coef /= sqrt(ADAG_scale_SUM + 0.000001);

				// calc delta
				scaleDelta = scaleDelta * ADAG_scale_Coef * coef;
				scaleDelta = this->limit_delta(scaleDelta);

				// update scale
				*(FILTER_batchN_scale_ptr + i) -= scaleDelta;
			}

		}// if batch norm
	}
   /////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// shift bias history
		for (int i = 0 ; i < output_dim[2]; i++)
		{
			// shift bias history
			this->shift_ADAG_History(CONV_bias_ADAG_history + (i*ADAG_HISTORY), ADAG_HISTORY);

			// set ADAG history to [0]
			float biasDelta = *(CONV_bias_delta_sum + i);
			*(CONV_bias_ADAG_history + (i*ADAG_HISTORY)) = biasDelta * biasDelta;
		}

		// shift kernel history
		for (int i = 0; i < kernel_size * kernel_size * input_dim[2] * output_dim[2]; i++)
		{
			// shift kernel history
			this->shift_ADAG_History(CONV_kernel_ADAG_hisroty + (i*ADAG_HISTORY), ADAG_HISTORY);

			// set ADAG history to [0]
			float kernelDelta = *(CONV_kernel_delta_sum + i);
			*(CONV_kernel_ADAG_hisroty + (i*ADAG_HISTORY)) = kernelDelta * kernelDelta;
		}

		////////////////////////////////////////////////////////////////////////

		// coef
		for (int i = 0; i < output_dim[2]; i++)
		{
			float bias_Delta = *(CONV_bias_delta_sum + i);
		
			float ADAG_bias_SUM = 0.0;
			float ADAG_bias_coef = INI_Coef;
			for (int a = 0; a < ADAG_HISTORY; a++)
			{
				ADAG_bias_SUM += *(CONV_bias_ADAG_history + (i*ADAG_HISTORY) + a);
			}

			ADAG_bias_coef /= sqrt(ADAG_bias_SUM + 0.000001);

			// calc delta
			bias_Delta = bias_Delta * ADAG_bias_coef * coef;
			bias_Delta = this->limit_delta(bias_Delta);

			// update bias
			*(CONV_bias_ptr + i) -= bias_Delta;
		}


		//
		for (int i = 0; i < kernel_size * kernel_size * input_dim[2] * output_dim[2]; i++)
		{
			float kernel_Delta = *(CONV_kernel_delta_sum + i);

			float ADAG_kernel_SUM = 0.0;
			float ADAG_kernel_coef = INI_Coef;
			for (int a = 0; a < ADAG_HISTORY; a++)
			{
				ADAG_kernel_SUM += *(CONV_kernel_ADAG_hisroty + (i*ADAG_HISTORY) + a);
			}

			ADAG_kernel_coef /= sqrt(ADAG_kernel_SUM + 0.000001);

			// calc delta
			kernel_Delta = kernel_Delta * ADAG_kernel_coef * coef;
			kernel_Delta = this->limit_delta(kernel_Delta);

			// update kernel
			*(CONV_kernel_ptr + i) -= kernel_Delta;
		}

	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
	//////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
	}// do nothing
}


void LayerClass::shift_ADAG_History(float* adagPtr, long num)
{
	for (int i = (num - 1); i > 0; i--)
	{
		// shift data
		*(adagPtr + i) = (*(adagPtr + (i - 1)))*0.8;
	}
}






////////////////////////////////////////////
//// Reset        //////////////////////////
////////////////////////////////////////////

void LayerClass::reset_weight()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		this->generate_AFFINE_initial_HX_weight();
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{} // do nothing......
	///////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		if (FILTER_mode == 0)
		{
			// set scale = 1.0, bias = 0.0
			for (int i = 0; i < output_dim[2]; i++)
			{
				*(FILTER_batchN_bias_ptr + i) = 0.0;
				*(FILTER_batchN_scale_ptr + i) = 1.0;
			}
		}
	}
	/////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		this->generate_initial_kernel_bias();
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
	//////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		this->set_DROPOUT_initial_gate();
	}
}



void LayerClass::clear_velocity_and_dropcount()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		long dataSize;
		// clear bias velocity
		for (int i = 0; i < output_dim[0]; i++)
		{
			*(AFFINE_bias_velocity + i) = 0.0;
		}

		// clear weight velocity
		for (int i = 0; i < input_dim[0] * output_dim[0]; i++)
		{
			*(AFFINE_weight_velocity + i) = 0.0;
		}
	}
	//////////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
	} // do nothing......
   ///////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		if (FILTER_mode == 0)
		{
			for (int i = 0; i < output_dim[2]; i++)
			{
				*(FILTER_batchN_bias_velocity + i) = 0.0;
				*(FILTER_batchN_scale_velocity + i) = 0.0;
			}
		}
	} 
   /////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		long dataSize;
		
		// clear kernel velocity
		for (int i = 0; i < (kernel_size * kernel_size * input_dim[0] * output_dim[2]); i++)
		{
			*(CONV_kernel_velocity + i) = 0.0;
		}

		// clear bias velocity
		for (int i = 0; i < output_dim[2]; i++)
		{
			*(CONV_bias_velocity + i) = 0.0;
		}
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
	//////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		// clear total count
		DROP_total_count = 0;

		// clear open count
		for (int i = 0; i < (input_dim[0] * input_dim[1] * input_dim[2] ); i++)
		{
			*(DROP_gate_open_count + i) = 0;
		}
	}
}





void LayerClass::clear_ADAG_history()
{
	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		for (int i = 0; i < output_dim[0] * ADAG_HISTORY; i++)
		{
			*(AFFINE_bias_ADAG_history + i) = 1.0;
		}
		for (int i = 0; i < input_dim[0] * output_dim[0] * ADAG_HISTORY; i++)
		{
			*(AFFINE_weight_ADAG_history + i) = 1.0;
		}
	}
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
	} // do nothing......
   ///////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		if (FILTER_mode == 0)
		{
			for (int i = 0; i < output_dim[2] * ADAG_HISTORY; i++)
			{
				*(FILTER_batchN_bias_ADAG_history + i) = 1.0;
				*(FILTER_batchN_scale_ADAG_history + i) = 1.0;
			}
		}
	}
	/////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		for (int i = 0; i < output_dim[2] * ADAG_HISTORY; i++)
		{
			*(CONV_bias_ADAG_history + i) = 1.0;
		}

		int KSIZE = kernel_size * kernel_size * input_dim[2] * output_dim[2];
		for (int i = 0; i < KSIZE * ADAG_HISTORY; i++)
		{
			*(CONV_kernel_ADAG_hisroty + i) = 1.0;
		}
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
	} // do nothing
	//////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
	}// do nothing
}




////////////////////////////////////////////
//// Save        //////////////////////////
////////////////////////////////////////////


void LayerClass::save_parameters_to_file(FILE* fpp)
{

	std::string paramStr;

	if (strcmp("AFFINE", layer_type_str) == 0)
	{
		// info for add layer AFFINE
		paramStr += std::to_string(input_dim[0]) + ";";
		paramStr += std::to_string(output_dim[0]) + ";";

		// bias
		for (int i = 0; i < output_dim[0]; i++)
		{
			float biasVal = *(AFFINE_bias_ptr + i);
			paramStr += std::to_string(biasVal) + "/";
		}

		// weight
		for (int i = 0; i < output_dim[0] * input_dim[0]; i++)
		{
			float weightVal = *(AFFINE_weight_ptr + i);
			paramStr += std::to_string(weightVal) + "/";
		}

		paramStr += "e\n";
	}
	///////////////////////////////////////////////////////
	else if (strcmp("CEMS", layer_type_str) == 0)
	{
		// info for add layer CEMS( inW, mode )
		paramStr += std::to_string(input_dim[0]) + ";";
		paramStr += std::to_string(CEMS_mode) + ";";

		paramStr += "no params to load.\n";
		paramStr += "e\n";
	}
   ///////////////////////////////////////////////////////
	else if (strcmp("FILTER", layer_type_str) == 0)
	{
		// info for add layer FILTER
		paramStr += std::to_string(input_dim[0]) + ";";
		paramStr += std::to_string(input_dim[1]) + ";";
		paramStr += std::to_string(input_dim[2]) + ";";
		paramStr += std::to_string(FILTER_mode) + ";";

		// scale
		for (int i = 0; i < output_dim[2]; i++)
		{
			float scaleVal = *(FILTER_batchN_scale_ptr + i);
			paramStr += std::to_string(scaleVal) + "/";
		}

		// bias
		for (int i = 0; i < output_dim[2]; i++)
		{
			float biasVal = *(FILTER_batchN_bias_ptr + i);
			paramStr += std::to_string(biasVal) + "/";
		}

		//paramStr += "no params to load.\n";
		paramStr += "e\n";
	}
   /////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", layer_type_str) == 0)
	{
		// info for add layer CONV
		paramStr += std::to_string(input_dim[0]) + ";";
		paramStr += std::to_string(input_dim[1]) + ";";
		paramStr += std::to_string(input_dim[2]) + ";";
		paramStr += std::to_string(kernel_size) + ";";
		paramStr += std::to_string(output_dim[2]) + ";";

		// bias
		for (int i = 0; i < output_dim[2]; i++)
		{
			float biasVal = *(CONV_bias_ptr + i);
			paramStr += std::to_string(biasVal) + "/";
		}

		int KSIZE = kernel_size * kernel_size * input_dim[2] * output_dim[2];
		for (int i = 0; i < KSIZE; i++)
		{
			float kernelVal = *(CONV_kernel_ptr + i);
			paramStr += std::to_string(kernelVal) + "/";
		}
		
		paramStr += "e\n";
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", layer_type_str) == 0)
	{
		// info for add layer POOLING
		paramStr += std::to_string(input_dim[0]) + ";";
		paramStr += std::to_string(input_dim[1]) + ";";
		paramStr += std::to_string(input_dim[2]) + ";";
		paramStr += std::to_string(POOLING_mode) + ";";

		paramStr += "no params to load.\n";
		paramStr += "e\n";
	}
	//////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", layer_type_str) == 0)
	{
		// info for add layer DROPOUT
		paramStr += std::to_string(input_dim[0]) + ";";
		paramStr += std::to_string(input_dim[1]) + ";";
		paramStr += std::to_string(input_dim[2]) + ";";

		// calculate gate val from gate counter
		int LEN = input_dim[0] * input_dim[1] * input_dim[2];
		float TOTAL_COUNT = (float)DROP_total_count;

		for (int i = 0; i < LEN; i++)
		{
			float openCount = (float)(*(DROP_gate_open_count + i));
			float rate = openCount / TOTAL_COUNT;

			paramStr += std::to_string(rate) + "/";
		}

		paramStr += "e\n";
	}

	// write to file
	fwrite(paramStr.c_str(), sizeof(char), paramStr.length(), fpp);
}



////////////////////////////////////////////////////////////
// called only when Act Use *******************************
////////////////////////////////////////////////////////////

void LayerClass::load_parameters_from_file(FILE* fpp, const char* Ltype)
{


	if (strcmp("AFFINE", Ltype) == 0)
	{
		// NUM_P = output_dim[0] + (output_dim[0] * input_dim[0])
		for (int b = 0; b < output_dim[0]; b++)
		{
			// get bias as char
			char biasChar[20];
			this->read_file_until(fpp, biasChar, '/');

			// convert to float
			float biasVal = atof(biasChar);

			// set
			*(AFFINE_bias_ptr + b) = biasVal;
		}

		for (int w = 0; w < output_dim[0] * input_dim[0]; w++)
		{
			// get weight as char
			char weightChar[20];
			this->read_file_until(fpp, weightChar, '/');

			// convert to float
			float weightVal = atof(weightChar);

			// set
			*(AFFINE_weight_ptr + w) = weightVal;
		}

		// just skip last "e"
		char find_e[10];
		this->read_file_until(fpp, find_e, 10);
	}
	///////////////////////////////////////////////////////
	else if (strcmp("CEMS", Ltype) == 0)
	{
		char no_params[30];
		char find_e[10];

		// just skip FILE
		this->read_file_until(fpp, no_params, 10); // 10 == \n
		this->read_file_until(fpp, find_e, 10);
	}
	///////////////////////////////////////////////////////
	else if (strcmp("FILTER", Ltype) == 0)
	{
		//char no_params[30];
		char find_e[10];

		// scale
		for (int s = 0; s < output_dim[2]; s++)
		{
			// get scale as char
			char scaleChar[20];
			this->read_file_until(fpp, scaleChar, '/');

			// convert to float
			float scaleVal = atof(scaleChar);
			
			// set
			*(FILTER_batchN_scale_ptr + s) = scaleVal;
		}

		// bias
		for (int b = 0; b < output_dim[2]; b++)
		{
			// get bias as char
			char biasChar[20];
			this->read_file_until(fpp, biasChar, '/');

			// convert to float
			float biasVal = atof(biasChar);

			// set
			*(FILTER_batchN_bias_ptr + b) = biasVal;
		}


		//this->read_file_until(fpp, no_params, 10); // 10 == \n
		this->read_file_until(fpp, find_e, 10);
	}
	/////////////////////////////////////////////////////////
	else if (strcmp("CONVOLUTION", Ltype) == 0)
	{
		// NUM_P = output_dim[2] + (KSIZE * KSIZE * input_dim[2] * output_dim[2])
		// bias
		for (int b = 0; b < output_dim[2]; b++)
		{
			// get bias as char
			char biasChar[20];
			this->read_file_until(fpp, biasChar, '/');

			// convert to float
			float biasVal = atof(biasChar);

			// set
			*(CONV_bias_ptr + b) = biasVal;
		}

		// kernel weight
		for (int k = 0; k < (kernel_size*kernel_size*input_dim[2] * output_dim[2]); k++)
		{
			// get kernel as char
			char kernelChar[20];
			this->read_file_until(fpp, kernelChar, '/');

			// convert to float
			float kernelVal = atof(kernelChar);

			// set
			*(CONV_kernel_ptr + k) = kernelVal;
		}

		// just skip last "e"
		char find_e[10];
		this->read_file_until(fpp, find_e, 10);
	}
	////////////////////////////////////////////////////////
	else if (strcmp("POOLING", Ltype) == 0)
	{
		char no_params[30];
		char find_e[10];

		// just skip FILE
		this->read_file_until(fpp, no_params, 10); // 10 == \n
		this->read_file_until(fpp, find_e, 10);
	}
	//////////////////////////////////////////////////////////////
	else if (strcmp("DROPOUT", Ltype) == 0)
	{
		// NUM_P == input_dim[0] * input_dim[1] * input_dim[2]

		for (int i = 0; i < input_dim[0] * input_dim[1] * input_dim[2]; i++)
		{
			// get dorpgate as char
			char dropChar[20];
			this->read_file_until(fpp, dropChar, '/');

			// convert to float
			float dropVal = atof(dropChar);

			// set
			*(DROP_gate_weight_ptr + i) = dropVal;
		}

		// just skip last "e"
		char find_e[10];
		this->read_file_until(fpp, find_e, 10);
	}
}







//::::::::::::::::::::::::::::::::::::::::::::
//// private method ::::::::::::::::::::::::::
//::::::::::::::::::::::::::::::::::::::::::::


float LayerClass::limit_delta(float value)
{
	float retVal = value;

	if (retVal > delta_limit_val)
	{
		retVal = delta_limit_val;
	}
	else if (retVal < -delta_limit_val)
	{
		retVal = delta_limit_val;
	}

	return retVal;
}




int LayerClass::find_divisor(int dim_val, int lim_val)
{
	int retVal = 1;

	// 2
	if ((dim_val % 2) == 0) { retVal = 2; }

	// 4
	if (lim_val == 4)
	{
		if ((dim_val % 4) == 0) { retVal = 4; }
	}

	// 8
	if (lim_val == 8)
	{
		if ((dim_val % 8) == 0) { retVal = 8; }
	}


	return retVal;
}


void LayerClass::read_file_until(FILE* fromFp, char* readBuf, char untilCode)
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

