#pragma once
#include "framework.h"
#include "OpenCL_Manager_Class.h"

#include <random>
#include <string>

#define WGT_XAVIER 0
#define WGT_HE 1
#define WGT_CONST 2

#define LOSS_MEAN_S 0
#define LOSS_CROSS_E 1
#define LOSS_MEAN_ABS 2 // not yet

#define LYR_BATCHNORM 0
#define LYR_SOFTMAX 1
#define LYR_ReLU 2
#define LYR_SIGMOID 3 // not yet
#define LYR_TANH 4 // not yet

#define POOL_MAX 0
#define POOL_AVERAGE 1 // not yet

#define DROP_TRAIN 0
#define DROP_ACT_USE 1 // not yet

#define KRN_3 3
#define KRN_5 5
#define KRN_7 7
#define KRN_9 9
#define KRN_11 11


#define ADAG_HISTORY 10

class LayerClass
{
public:
	LayerClass();
	~LayerClass();

	// COMMON
	char layer_type_str[32];
	OpenCL_Manager_Class* cl_obj;

	// variables
	int DEVICE_id = 0;
	int num_CPU_img = 0;
	int num_GPU_img = 0;
	int answer_width = 0;
	int input_dim[3] = {1,1,1};
	int output_dim[3] = { 1,1,1 };

	// for recording
	int layer_number = 0;
	long num_saved_param = 0; // not used currently


	// COMMON data
	float* input_data_ptr;
	float* output_data_ptr;
	float* back_in_ptr; // set by prev
	float* back_out_ptr;
	float* answer_ptr; // used only in last layer
	float weight_decay_val = 0.0;
	float weight_decay_sum = 0.0;
	float delta_limit_val = 0.1;
	int LOCAL_IMG = 1;

	cl_mem mem_input_data = NULL; // set by prev
	cl_mem mem_output_data = NULL;
	cl_mem mem_back_in = NULL; // set by prev
	cl_mem mem_back_out = NULL;
	cl_mem mem_answer = NULL; // used only in last layer


	// add layer
	void init_as_AFFINE(int inW, int outW, int iniWeight);
	void init_as_CEMS(int inW, int mode);
	void init_as_FILTER(int inX, int inY, int inZ, int mode);
	void init_as_CONVOLUTION(int inX, int inY, int inZ, int krnSize, int outZ, int weightMode);
	void init_as_MAX_POOLING(int inX, int inY, int inZ, int poolMode);
	void init_as_DROPOUT(int inX, int inY, int inZ, float RATE, int dropMode);

	// setup method
	void setup_cl_mem();
	void setup_cl_kernel();
	//CPU process
	void learn_CPU(float* inPtr, float* ansPtr);
	void back_propagation_CPU(float* backInPtr);
	// GPU process
	void learn_CL();
	void back_propagation_CL();
	void readback_CL();
	// modify
	void update_by_SDG(float coef);
	void update_by_Momemtom(float coef, float brake);
	void update_by_AdaGrad(float coef);
	// update method
	void copy_weight_bias_from(LayerClass* srcLayer);
	void average_weight_bias_lossSum(float coef);
	void change_dropout_gate();
	void weight_decay(float decayCoef);
	void update_cl_mem_weight();
	// clear
	void clear_sum();
	// reset
	void reset_weight();
	void clear_velocity_and_dropcount();
	void clear_ADAG_history();
	// save & load file
	void save_parameters_to_file(FILE* fpp);
	void load_parameters_from_file(FILE* fpp, const char* Ltype);


private:
	// utility method
	float limit_delta(float value);
	void shift_ADAG_History(float* adagPtr, long num);
	int find_divisor(int dim_val, int lim_val);
	void read_file_until(FILE* fromFp, char* readBuf, char untilCode);

	//////// AFFINE ////////////////////////////////////////////
	//////// AFFINE ////////////////////////////////////////////
	//////// AFFINE ////////////////////////////////////////////
public:
	int AFFINE_weight_mode = 0;

	// data
	float* AFFINE_bias_ptr;
	float* AFFINE_weight_ptr;
	float* AFFINE_bias_delta_sum; // cleared at every loop end
	float* AFFINE_weight_delta_sum; // cleared at every loop end
	float* AFFINE_bias_velocity; // cleared by reset
	float* AFFINE_weight_velocity; // cleared by reset

	float* AFFINE_bias_ADAG_history; // cleared by reset
	float* AFFINE_weight_ADAG_history; // cleared by reset


	cl_mem mem_AFFINE_bias; // no image layer
	cl_mem mem_AFFINE_weight; // no image layer
	cl_mem mem_AFFINE_bias_delta_sum; // no image layer
	cl_mem mem_AFFINE_weight_delta_sum; // no image layer

	// method
	// setup
	void alloc_AFFINE_CPU_memory();
	void generate_AFFINE_initial_HX_weight();
	void setup_AFFINE_cl_mem();
	void setup_AFFINE_cl_kernel();
	// CPU process
	void learn_CPU_AFFINE(float* inPtr, float* ansPtr);
	void back_propagation_AFFINE(float* backInPtr);
	// GPU process
	void enqueue_forward_kernel_AFFINE();
	void enqueue_back_kernel_AFFINE();
	void enqueue_read_back_kernel_AFFINE();

	// kernel
	cl_kernel krn_AFFINE;
	cl_kernel krn_AFFINE_back_biasSum;
	cl_kernel krn_AFFINE_back_weightSum;
	cl_kernel krn_AFFINE_back_out;


//////// CEMS ////////////////////////////////////////////
//////// CEMS ////////////////////////////////////////////
//////// CEMS ////////////////////////////////////////////

	int CEMS_mode = 0;
	
	// data
	float* CEMS_loss_sum_ptr;

	cl_mem mem_CEMS_loss_sum;

	// method
	// setup
	void alloc_CEMS_CPU_memory();
	void setup_CEMS_cl_mem();
	void setup_CEMS_cl_kernel();
	// CPU process
	void learn_CPU_CEMS(float* inPtr, float* ansPtr);
	// back_proagation is done in forward process
	
	// GPU process
	void enqueue_forward_kernel_CEMS();
	void enqueue_back_kernel_CEMS();
	void enqueue_read_back_CEMS();

	// modify process
	void add_weight_bias_lossSum(LayerClass* srcLayer);


	// kernel
	cl_kernel krn_CEMS_meanS;
	cl_kernel krn_CEMS_crossE;
	cl_kernel krn_CEMS_back_lossSum;


//////// FILTER ////////////////////////////////////////////
//////// FILTER ////////////////////////////////////////////
//////// FILTER ////////////////////////////////////////////

	int FILTER_mode = 0;
	float* FILTER_batchN_scale_ptr;
	float* FILTER_batchN_bias_ptr;
	float* FILTER_batchN_hold_midVal;
	float* FILTER_batchN_scale_delta_sum;
	float* FILTER_batchN_bias_delta_sum;
	float* FILTER_batchN_scale_velocity;
	float* FILTER_batchN_bias_velocity;
	float* FILTER_batchN_scale_ADAG_history;
	float* FILTER_batchN_bias_ADAG_history;

	float* FILTER_hold_DEVI_ptr;
	float* FILTER_hold_DEVI_each_ptr;
	float* FILTER_hold_exp;
	float* FILTER_hold_expSum;

	cl_mem mem_FILTER_batchN_scale;
	cl_mem mem_FILTER_batchN_bias;
	cl_mem mem_FILTER_batchN_hold_midVal;
	cl_mem mem_FILTER_batchN_scale_delta_each;
	cl_mem mem_FILTER_batchN_bias_delta_each;
	cl_mem mem_FILTER_batchN_scale_delta_sum;
	cl_mem mem_FILTER_batchN_bias_delta_sum;

	cl_mem mem_FILTER_hold_DEVI;
	cl_mem mem_FILTER_hold_DEVI_each;
	cl_mem mem_FILTER_temp;
	cl_mem mem_FILTER_hold_exp;
	cl_mem mem_FILTER_hold_expSum;

	// method
	// setup
	void alloc_FILTER_CPU_memory();
	void setup_FILTER_cl_mem();
	void setup_FILTER_cl_kernel();
	// CPU process
	void learn_CPU_FILTER(float* inPtr, float* ansPtr);
	void back_propagation_FILTER(float* backInPtr);

	void FILTER_batch_norm(float* inPtr);
	void FILTER_softmax(float* inPtr);
	void FILTER_ReLU(float* inPtr);

	void FILTER_back_batch_norm(float* backInPtr);
	void FILTER_back_softmax(float* backInPtr);
	void FILTER_back_ReLU(float* backInPtr);

	// GPU process
	void enqueue_forward_kernel_FILTER();
	void enqueue_back_kernel_FILTER();
	void enqueue_read_back_FILTER();

	// kernel
	cl_kernel krn_FILTER_batch_norm;
	cl_kernel krn_FILTER_batch_norm_each_z;
	cl_kernel krn_FILTER_softmax;
	cl_kernel krn_FILTER_ReLU;

	cl_kernel krn_FILTER_back_batch_norm;
	cl_kernel krn_FILTER_back_batch_norm_each_z;
	cl_kernel krn_FILTER_back_batch_norm_sumDelta;
	cl_kernel krn_FILTER_back_softmax;
	cl_kernel krn_FILTER_back_ReLU;


//////// CONVOLUTION ////////////////////////////////////////////
//////// CONVOLUTION ////////////////////////////////////////////
//////// CONVOLUTION ////////////////////////////////////////////

	int CONV_weight_mode = 0;
	int kernel_size = 0;
	float* CONV_kernel_ptr;
	float* CONV_bias_ptr;
	float* CONV_kernel_velocity;
	float* CONV_bias_velocity;
	float* CONV_kernel_delta_sum;
	float* CONV_bias_delta_sum;

	float* CONV_kernel_ADAG_hisroty;
	float* CONV_bias_ADAG_history;


	cl_mem mem_CONV_pad_input;
	cl_mem mem_CONV_pad_back_in;
	cl_mem mem_CONV_kernel;
	cl_mem mem_CONV_bias;
	cl_mem mem_CONV_kernel_delta_each;
	cl_mem mem_CONV_bias_delta_each;
	cl_mem mem_CONV_kernel_delta_sum;
	cl_mem mem_CONV_bias_delta_sum;


	// method
	// setup
	void alloc_CONV_CPU_memory();
	void generate_initial_kernel_bias();
	void setup_CONV_cl_mem();
	void setup_CONV_cl_kernel();
	// CPU process
	void learn_CPU_CONVOLUTION(float* inPtr, float* ansPtr);
	void back_propagation_CONVOLUTION(float* backInPtr);
	void CONV_back_bias_sum();
	void CONV_back_kernel_sum();
	void CONV_back_out();
	// GPU process
	void enqueue_forward_kernel_CONVOLUTION();
	void enqueue_back_kernel_CONVOLUTION();
	void enqueue_read_back_CONVOLUTION();

	// kernel
	cl_kernel krn_CONV_copy_pad_input;
	cl_kernel krn_CONV_copy_pad_back_in;

	cl_kernel krn_CONV_convolution_test;

	cl_kernel krn_CONV_convolution;
	cl_kernel krn_CONV_back_bias_each;
	cl_kernel krn_CONV_back_kernel_each;
	cl_kernel krn_CONV_sum_back_bias;
	cl_kernel krn_CONV_sum_back_kernel;

	cl_kernel krn_CONV_back_out_test;

	cl_kernel krn_CONV_back_out;



//////// POOLING ////////////////////////////////////////////
//////// POOLING ////////////////////////////////////////////
//////// POOLING ////////////////////////////////////////////

	int POOLING_mode = 0; // 0 max, 1 average,
	float* POOL_selected_mask_ptr;

	cl_mem mem_POOL_selected_mask;

	// setup
	void alloc_POOL_CPU_memory();
	void setup_POOL_cl_mem();
	void setup_POOL_cl_kernel();
	// CPU process
	void learn_CPU_POOLING(float* inPtr, float* ansPtr);
	void back_propagation_POOLING(float* backInPtr);
	// GPU process
	void enqueue_forward_kernel_POOLING();
	void enqueue_back_kernel_POOLING();

	// kernel
	cl_kernel krn_POOL_max_pooling;
	cl_kernel krn_POOL_back_max_pooling;



//////// DROPOUT ////////////////////////////////////////////
//////// DROPOUT ////////////////////////////////////////////
//////// DROPOUT ////////////////////////////////////////////

	int DROPOUT_mode = 0; // training or actual use
	float DROPOUT_rate = 0.0;
	float* DROP_gate_weight_ptr; // 0.0 or 1.0 when training, 0.0 ~ 1.0 when act use.
	unsigned int* DROP_gate_open_count; // must be reset
	unsigned int DROP_total_count = 0; // must be reset

	cl_mem mem_DROP_gate_weight; // no image layer

	// setup
	void alloc_DROPOUT_CPU_memory();
	void set_DROPOUT_initial_gate();
	void setup_DROPOUT_cl_mem();
	void setup_DROPOUT_cl_kernel();
	// CPU process
	void learn_CPU_DROPOUT(float* inPtr, float* ansPtr);
	void back_propagation_DROPOUT(float* backInPtr);
	// GPU process
	void enqueue_forward_kernel_DROPOUT();
	void enqueue_back_kernel_DROPOUT();

	// kernel
	cl_kernel krn_DROP_dropout;
	cl_kernel krn_DROP_back_dropout;
};

