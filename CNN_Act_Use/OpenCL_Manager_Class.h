#pragma once

#include "framework.h"

// OpenCL
#pragma comment(lib, "CL/x64/OpenCL.lib")
//#pragma comment(lib, "CL/x86/OpenCL.lib")
#include "CL/cl.h"

#include <fstream>

#define NUM_MAX_CL_DEVICE 64

class OpenCL_Manager_Class
{
public:
	cl_platform_id platformID[64];
	cl_uint num_platform;
	///////////////////////////////
	cl_device_id GPUdeviceID_array[NUM_MAX_CL_DEVICE];
	cl_uint num_devices[64];
	cl_uint num_total_GPU_devices = 0;
	///////////////////////////////
	cl_context cl_CTX_obj[NUM_MAX_CL_DEVICE];
	cl_command_queue cl_CMQ_obj[NUM_MAX_CL_DEVICE];
	cl_program cl_PRG_obj[32];



	OpenCL_Manager_Class();
	~OpenCL_Manager_Class();

	///////// set ///////////////////
	void setup_openCL();
	void create_program_from_file(int deviceID, const char* filePath);

	// utility
	void create_mem_util(cl_int deviceIDX, cl_mem* memPtr, long dataSize, cl_int flag);
	void update_mem_contents_util(cl_int deviceIDX, cl_mem* memPtr, float* srcPtr, long dataSize);
	// kernel utility
	void create_kernel_util(cl_int deviceIDX, cl_kernel* krnPtr, const char* fname);
	void set_kernel_arg_mem(cl_kernel* krnPtr, int argIDX, cl_mem* memPtr);
	void set_kernel_arg_int_val(cl_kernel* krnPtr, int argIDX, cl_int val);
	void set_kernel_arg_float_local(cl_kernel* krnPtr, int argIDX, cl_uint size);

private:
	/////////////////////////////////
	void setup_platform();
	void setup_gpu_device();
	void get_device_info_func(int devIDX, int ENUM, long size, void* ptr);
	void create_context_and_queue();
	////////////////////////////
};

