/*******************************************************************************
 * Copyright (c) 2008-2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

/* $Revision: 11928 $ on $Date: 2010-07-13 09:04:56 -0700 (Tue, 13 Jul 2010) $ */

/* cl_ext.h contains OpenCL extensions which don't have external */
/* (OpenGL, D3D) dependencies.                                   */

#ifndef __CL_EXT_H
#define __CL_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __APPLE__
        #include "OpenCL/cl.h"
    #include "AvailabilityMacros.h"
#else
        #include "CL/cl.h"
#endif

/* cl_khr_fp16 extension - no extension #define since it has no functions  */
#define CL_DEVICE_HALF_FP_CONFIG                    0x1033

/* Memory object destruction
 *
 * Apple extension for use to manage externally allocated buffers used with cl_mem objects with CL_MEM_USE_HOST_PTR
 *
 * Registers a user callback function that will be called when the memory object is deleted and its resources 
 * freed. Each call to clSetMemObjectCallbackFn registers the specified user callback function on a callback 
 * stack associated with memobj. The registered user callback functions are called in the reverse order in 
 * which they were registered. The user callback functions are called and then the memory object is deleted 
 * and its resources freed. This provides a mechanism for the application (and libraries) using memobj to be 
 * notified when the memory referenced by host_ptr, specified when the memory object is created and used as 
 * the storage bits for the memory object, can be reused or freed.
 *
 * The application may not call CL api's with the cl_mem object passed to the pfn_notify.
 *
 * Please check for the "cl_APPLE_SetMemObjectDestructor" extension using clGetDeviceInfo(CL_DEVICE_EXTENSIONS)
 * before using.
 */
#define cl_APPLE_SetMemObjectDestructor 1
cl_int  CL_API_ENTRY clSetMemObjectDestructorAPPLE(  cl_mem /* memobj */, 
                                        void (* /*pfn_notify*/)( cl_mem /* memobj */, void* /*user_data*/), 
                                        void * /*user_data */ )             CL_EXT_SUFFIX__VERSION_1_0;  


/* Context Logging Functions
 *
 * The next three convenience functions are intended to be used as the pfn_notify parameter to clCreateContext().
 * Please check for the "cl_APPLE_ContextLoggingFunctions" extension using clGetDeviceInfo(CL_DEVICE_EXTENSIONS)
 * before using.
 *
 * clLogMessagesToSystemLog fowards on all log messages to the Apple System Logger 
 */
#define cl_APPLE_ContextLoggingFunctions 1
extern void CL_API_ENTRY clLogMessagesToSystemLogAPPLE(  const char * /* errstr */, 
                                            const void * /* private_info */, 
                                            size_t       /* cb */, 
                                            void *       /* user_data */ )  CL_EXT_SUFFIX__VERSION_1_0;

/* clLogMessagesToStdout sends all log messages to the file descriptor stdout */
extern void CL_API_ENTRY clLogMessagesToStdoutAPPLE(   const char * /* errstr */, 
                                          const void * /* private_info */, 
                                          size_t       /* cb */, 
                                          void *       /* user_data */ )    CL_EXT_SUFFIX__VERSION_1_0;

/* clLogMessagesToStderr sends all log messages to the file descriptor stderr */
extern void CL_API_ENTRY clLogMessagesToStderrAPPLE(   const char * /* errstr */, 
                                          const void * /* private_info */, 
                                          size_t       /* cb */, 
                                          void *       /* user_data */ )    CL_EXT_SUFFIX__VERSION_1_0;


/************************ 
* cl_khr_icd extension *                                                  
************************/
#define cl_khr_icd 1

/* cl_platform_info                                                        */
#define CL_PLATFORM_ICD_SUFFIX_KHR                  0x0920

/* Additional Error Codes                                                  */
#define CL_PLATFORM_NOT_FOUND_KHR                   -1001

extern CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint          /* num_entries */,
                       cl_platform_id * /* platforms */,
                       cl_uint *        /* num_platforms */);

typedef CL_API_ENTRY cl_int (CL_API_CALL *clIcdGetPlatformIDsKHR_fn)(
    cl_uint          /* num_entries */,
    cl_platform_id * /* platforms */,
    cl_uint *        /* num_platforms */);


/* Extension: cl_khr_image2D_buffer
 *
 * This extension allows a 2D image to be created from a cl_mem buffer without a copy.
 * The type associated with a 2D image created from a buffer in an OpenCL program is image2d_t.
 * Both the sampler and sampler-less read_image built-in functions are supported for 2D images
 * and 2D images created from a buffer.  Similarly, the write_image built-ins are also supported
 * for 2D images created from a buffer.
 *
 * When the 2D image from buffer is created, the client must specify the width,
 * height, image format (i.e. channel order and channel data type) and optionally the row pitch
 *
 * The pitch specified must be a multiple of CL_DEVICE_IMAGE_PITCH_ALIGNMENT pixels.
 * The base address of the buffer must be aligned to CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT pixels.
 */
    
/*************************************
 * cl_khr_initalize_memory extension *
 *************************************/
    
#define CL_CONTEXT_MEMORY_INITIALIZE_KHR            0x2030
    
    
/**************************************
 * cl_khr_terminate_context extension *
 **************************************/
    
#define CL_DEVICE_TERMINATE_CAPABILITY_KHR          0x2031
#define CL_CONTEXT_TERMINATE_KHR                    0x2032

#define cl_khr_terminate_context 1
extern CL_API_ENTRY cl_int CL_API_CALL clTerminateContextKHR(cl_context /* context */) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clTerminateContextKHR_fn)(cl_context /* context */) CL_EXT_SUFFIX__VERSION_1_2;
    
    
/*
 * Extension: cl_khr_spir
 *
 * This extension adds support to create an OpenCL program object from a 
 * Standard Portable Intermediate Representation (SPIR) instance
 */

#define CL_DEVICE_SPIR_VERSIONS                     0x40E0
#define CL_PROGRAM_BINARY_TYPE_INTERMEDIATE         0x40E1


/******************************************
* cl_nv_device_attribute_query extension *
******************************************/
/* cl_nv_device_attribute_query extension - no extension #define since it has no functions */
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV            0x4002
#define CL_DEVICE_WARP_SIZE_NV                      0x4003
#define CL_DEVICE_GPU_OVERLAP_NV                    0x4004
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
#define CL_DEVICE_INTEGRATED_MEMORY_NV              0x4006

/*********************************
* cl_amd_device_attribute_query *
*********************************/
#define CL_DEVICE_PROFILING_TIMER_OFFSET_AMD        0x4036

/*********************************
* cl_arm_printf extension
*********************************/
#define CL_PRINTF_CALLBACK_ARM                      0x40B0
#define CL_PRINTF_BUFFERSIZE_ARM                    0x40B1

#ifdef _WIN32
#if defined CL_DX9_MEDIA_SHARING_INTEL_EXT

#include <d3d9.h>
#include <dxvahd.h>
#include <wtypes.h>
#include <d3d9types.h>
 
/******************************************************************************/    
/* cl_intel_dx9_media_sharing extension                                       */
#define cl_intel_dx9_media_sharing 1

typedef cl_uint cl_dx9_device_source_intel;
typedef cl_uint cl_dx9_device_set_intel;

/******************************************************************************/

// Error Codes
#define CL_INVALID_DX9_DEVICE_INTEL                   -1010
#define CL_INVALID_DX9_RESOURCE_INTEL                 -1011
#define CL_DX9_RESOURCE_ALREADY_ACQUIRED_INTEL        -1012
#define CL_DX9_RESOURCE_NOT_ACQUIRED_INTEL            -1013

// cl_dx9_device_source_intel
#define CL_D3D9_DEVICE_INTEL                          0x4022
#define CL_D3D9EX_DEVICE_INTEL                        0x4070
#define CL_DXVA_DEVICE_INTEL                          0x4071

// cl_dx9_device_set_intel
#define CL_PREFERRED_DEVICES_FOR_DX9_INTEL            0x4024
#define CL_ALL_DEVICES_FOR_DX9_INTEL                  0x4025

// cl_context_info
#define CL_CONTEXT_D3D9_DEVICE_INTEL                  0x4026
#define CL_CONTEXT_D3D9EX_DEVICE_INTEL                0x4072
#define CL_CONTEXT_DXVA_DEVICE_INTEL                  0x4073

// cl_mem_info
#define CL_MEM_DX9_RESOURCE_INTEL                     0x4027
#define CL_MEM_DX9_SHARED_HANDLE_INTEL                0x4074

// cl_image_info
#define CL_IMAGE_DX9_PLANE_INTEL                      0x4075

// cl_command_type
#define CL_COMMAND_ACQUIRE_DX9_OBJECTS_INTEL          0x402A
#define CL_COMMAND_RELEASE_DX9_OBJECTS_INTEL          0x402B

//packed YUV channel order
#define CL_YUYV_INTEL                                 0x4076
#define CL_UYVY_INTEL                                 0x4077
#define CL_YVYU_INTEL                                 0x4078
#define CL_VYUY_INTEL                                 0x4079

/******************************************************************************/

typedef CL_API_ENTRY cl_int (CL_API_CALL* clGetDeviceIDsFromDX9INTEL_fn)(
    cl_platform_id              /*platform*/,
    cl_dx9_device_source_intel  /*dx9_device_source*/,
    void*                       /*dx9_object*/,
    cl_dx9_device_set_intel     /*dx9_device_set*/,
    cl_uint                     /*num_entries*/, 
    cl_device_id*               /*devices*/, 
    cl_uint*                    /*num_devices*/);
    
typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromDX9MediaSurfaceINTEL_fn)(
    cl_context                  /*context*/,
    cl_mem_flags                /*flags*/,
    IDirect3DSurface9 *         /*resource*/,
    HANDLE                      /*sharedHandle*/,
    UINT                        /*plane*/,
    cl_int *                    /*errcode_ret*/);
    
typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueAcquireDX9ObjectsINTEL_fn)(
    cl_command_queue            /*command_queue*/,
    cl_uint                     /*num_objects*/,
    const cl_mem *              /*mem_objects*/,
    cl_uint                     /*num_events_in_wait_list*/,
    const cl_event *            /*event_wait_list*/,
    cl_event *                  /*event*/);
    
typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueReleaseDX9ObjectsINTEL_fn)(
    cl_command_queue            /*command_queue*/,
    cl_uint                     /*num_objects*/,
    cl_mem *                    /*mem_objects*/,
    cl_uint                     /*num_events_in_wait_list*/,
    const cl_event *            /*event_wait_list*/,
    cl_event *                  /*event*/);

#endif // CL_DX9_MEDIA_SHARING_INTEL_EXT
#endif

/*********************************
* cl_intel_accelerator extension *
*********************************/
#define cl_intel_accelerator 1
#define cl_intel_motion_estimation 1

typedef struct _cl_accelerator_intel*     cl_accelerator_intel;
typedef cl_uint                           cl_accelerator_type_intel;
typedef cl_uint                           cl_accelerator_info_intel;

typedef struct _cl_motion_estimation_desc_intel {
    cl_uint mb_block_type;
    cl_uint subpixel_mode;
    cl_uint sad_adjust_mode;
    cl_uint search_path_type;
} cl_motion_estimation_desc_intel;

/* Error Codes */
#define CL_INVALID_ACCELERATOR_INTEL            -1094
#define CL_INVALID_ACCELERATOR_TYPE_INTEL       -1095
#define CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL -1096
#define CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL -1097

/* Deprecated Error Codes */
#define CL_INVALID_ACCELERATOR_INTEL_DEPRECATED            -6000
#define CL_INVALID_ACCELERATOR_TYPE_INTEL_DEPRECATED       -6001
#define CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL_DEPRECATED -6002
#define CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL_DEPRECATED -6003

/* cl_accelerator_type_intel */
#define CL_ACCELERATOR_TYPE_MOTION_ESTIMATION_INTEL     0x0

/* cl_accelerator_info_intel */
#define CL_ACCELERATOR_DESCRIPTOR_INTEL                 0x4090
#define CL_ACCELERATOR_REFERENCE_COUNT_INTEL            0x4091
#define CL_ACCELERATOR_CONTEXT_INTEL                    0x4092
#define CL_ACCELERATOR_TYPE_INTEL                       0x4093

/*cl_motion_detect_desc_intel flags */
#define CL_ME_MB_TYPE_16x16_INTEL                       0x0
#define CL_ME_MB_TYPE_8x8_INTEL                         0x1
#define CL_ME_MB_TYPE_4x4_INTEL                         0x2

#define CL_ME_SUBPIXEL_MODE_INTEGER_INTEL               0x0
#define CL_ME_SUBPIXEL_MODE_HPEL_INTEL                  0x1
#define CL_ME_SUBPIXEL_MODE_QPEL_INTEL                  0x2

#define CL_ME_SAD_ADJUST_MODE_NONE_INTEL                0x0
#define CL_ME_SAD_ADJUST_MODE_HAAR_INTEL                0x1

#define CL_ME_SEARCH_PATH_RADIUS_2_2_INTEL              0x0
#define CL_ME_SEARCH_PATH_RADIUS_4_4_INTEL              0x1
#define CL_ME_SEARCH_PATH_RADIUS_16_12_INTEL            0x5

extern CL_API_ENTRY cl_accelerator_intel CL_API_CALL
clCreateAcceleratorINTEL(
    cl_context                  /* context */,
    cl_accelerator_type_intel   /* accelerator_type */,
    size_t                      /* descriptor_size */,
    const void*                 /* descriptor */,
    cl_int*                     /* errcode_ret */ ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_accelerator_intel
    (CL_API_CALL *clCreateAcceleratorINTEL_fn)(
    cl_context                  /* context */,
    cl_accelerator_type_intel   /* accelerator_type */,
    size_t                      /* descriptor_size */,
    const void*                 /* descriptor */,
    cl_int*                     /* errcode_ret */ ) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetAcceleratorInfoINTEL
(
    cl_accelerator_intel        /* accelerator */,
    cl_accelerator_info_intel   /* param_name */,
    size_t                      /* param_value_size */,
    void*                       /* param_value */,
    size_t*                     /* param_value_size_ret */ ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
    (CL_API_CALL *clGetAcceleratorInfoINTEL_fn)(
    cl_accelerator_intel        /* accelerator */,
    cl_accelerator_info_intel   /* param_name */,
    size_t                      /* param_value_size */,
    void*                       /* param_value */,
    size_t*                     /* param_value_size_ret */ ) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clSetAcceleratorInfoINTEL
(
    cl_accelerator_intel        /* accelerator */,
    cl_accelerator_info_intel   /* param_name */,
    size_t                      /* param_value_size */,
    void*                       /* param_value */ ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
    (CL_API_CALL *clSetAcceleratorInfoINTEL_fn)(
    cl_accelerator_intel        /* accelerator */,
    cl_accelerator_info_intel   /* param_name */,
    size_t                      /* param_value_size */,
    void*                       /* param_value */ ) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainAcceleratorINTEL(
    cl_accelerator_intel        /* accelerator */ ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
    (CL_API_CALL *clRetainAcceleratorINTEL_fn)(
    cl_accelerator_intel        /* accelerator */ ) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseAcceleratorINTEL(
    cl_accelerator_intel        /* accelerator */ ) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
    (CL_API_CALL *clReleaseAcceleratorINTEL_fn)(
    cl_accelerator_intel        /* accelerator */ ) CL_EXT_SUFFIX__VERSION_1_2;

#ifdef CL_VERSION_1_1
   /***********************************
    * cl_ext_device_fission extension *
    ***********************************/
    #define cl_ext_device_fission   1
    
    extern CL_API_ENTRY cl_int CL_API_CALL
    clReleaseDeviceEXT( cl_device_id /*device*/ ) CL_EXT_SUFFIX__VERSION_1_1; 
    
    typedef CL_API_ENTRY cl_int 
    (CL_API_CALL *clReleaseDeviceEXT_fn)( cl_device_id /*device*/ ) CL_EXT_SUFFIX__VERSION_1_1;

    extern CL_API_ENTRY cl_int CL_API_CALL
    clRetainDeviceEXT( cl_device_id /*device*/ ) CL_EXT_SUFFIX__VERSION_1_1; 
    
    typedef CL_API_ENTRY cl_int 
    (CL_API_CALL *clRetainDeviceEXT_fn)( cl_device_id /*device*/ ) CL_EXT_SUFFIX__VERSION_1_1;

    typedef cl_ulong  cl_device_partition_property_ext;
    extern CL_API_ENTRY cl_int CL_API_CALL
    clCreateSubDevicesEXT(  cl_device_id /*in_device*/,
                            const cl_device_partition_property_ext * /* properties */,
                            cl_uint /*num_entries*/,
                            cl_device_id * /*out_devices*/,
                            cl_uint * /*num_devices*/ ) CL_EXT_SUFFIX__VERSION_1_1;

    typedef CL_API_ENTRY cl_int 
    ( CL_API_CALL * clCreateSubDevicesEXT_fn)(  cl_device_id /*in_device*/,
                                                const cl_device_partition_property_ext * /* properties */,
                                                cl_uint /*num_entries*/,
                                                cl_device_id * /*out_devices*/,
                                                cl_uint * /*num_devices*/ ) CL_EXT_SUFFIX__VERSION_1_1;

    /* cl_device_partition_property_ext */
    #define CL_DEVICE_PARTITION_EQUALLY_EXT             0x4050
    #define CL_DEVICE_PARTITION_BY_COUNTS_EXT           0x4051
    #define CL_DEVICE_PARTITION_BY_NAMES_EXT            0x4052
    #define CL_DEVICE_PARTITION_BY_NAMES_INTEL          0x4052
    #define CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT  0x4053
    
    /* clDeviceGetInfo selectors */
    #define CL_DEVICE_PARENT_DEVICE_EXT                 0x4054
    #define CL_DEVICE_PARTITION_TYPES_EXT               0x4055
    #define CL_DEVICE_AFFINITY_DOMAINS_EXT              0x4056
    #define CL_DEVICE_REFERENCE_COUNT_EXT               0x4057
    #define CL_DEVICE_PARTITION_STYLE_EXT               0x4058
    
    /* error codes */
    #define CL_DEVICE_PARTITION_FAILED_EXT              -1057
    #define CL_INVALID_PARTITION_COUNT_EXT              -1058
    #define CL_INVALID_PARTITION_NAME_EXT               -1059
    
    /* CL_AFFINITY_DOMAINs */
    #define CL_AFFINITY_DOMAIN_L1_CACHE_EXT             0x1
    #define CL_AFFINITY_DOMAIN_L2_CACHE_EXT             0x2
    #define CL_AFFINITY_DOMAIN_L3_CACHE_EXT             0x3
    #define CL_AFFINITY_DOMAIN_L4_CACHE_EXT             0x4
    #define CL_AFFINITY_DOMAIN_NUMA_EXT                 0x10
    #define CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT     0x100
    
    /* cl_device_partition_property_ext list terminators */
    #define CL_PROPERTIES_LIST_END_EXT                  ((cl_device_partition_property_ext) 0)
    #define CL_PARTITION_BY_COUNTS_LIST_END_EXT         ((cl_device_partition_property_ext) 0)
    #define CL_PARTITION_BY_NAMES_LIST_END_EXT          ((cl_device_partition_property_ext) 0 - 1)
    #define CL_PARTITION_BY_NAMES_LIST_END_INTEL        ((cl_device_partition_property_ext) 0 - 1)

    #define CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL      (1 << 31)


/*********************************
* cl_qcom_ext_host_ptr extension
*********************************/

#define CL_MEM_EXT_HOST_PTR_QCOM                  (1 << 29)

#define CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM   0x40A0      
#define CL_DEVICE_PAGE_SIZE_QCOM                  0x40A1
#define CL_IMAGE_ROW_ALIGNMENT_QCOM               0x40A2
#define CL_IMAGE_SLICE_ALIGNMENT_QCOM             0x40A3
#define CL_MEM_HOST_UNCACHED_QCOM                 0x40A4
#define CL_MEM_HOST_WRITEBACK_QCOM                0x40A5
#define CL_MEM_HOST_WRITETHROUGH_QCOM             0x40A6
#define CL_MEM_HOST_WRITE_COMBINING_QCOM          0x40A7

typedef cl_uint                                   cl_image_pitch_info_qcom;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceImageInfoQCOM(cl_device_id             device,
                         size_t                   image_width,
                         size_t                   image_height,
                         const cl_image_format   *image_format,
                         cl_image_pitch_info_qcom param_name,
                         size_t                   param_value_size,
                         void                    *param_value,
                         size_t                  *param_value_size_ret);

typedef struct _cl_mem_ext_host_ptr
{
    /* Type of external memory allocation. */
    /* Legal values will be defined in layered extensions. */
    cl_uint  allocation_type;
            
    /* Host cache policy for this external memory allocation. */
    cl_uint  host_cache_policy;

} cl_mem_ext_host_ptr;

/*********************************
* cl_qcom_ion_host_ptr extension
*********************************/

#define CL_MEM_ION_HOST_PTR_QCOM                  0x40A8

typedef struct _cl_mem_ion_host_ptr
{
    /* Type of external memory allocation. */
    /* Must be CL_MEM_ION_HOST_PTR_QCOM for ION allocations. */
    cl_mem_ext_host_ptr  ext_host_ptr;

    /* ION file descriptor */
    int                  ion_filedesc;
            
    /* Host pointer to the ION allocated memory */
    void*                ion_hostptr;

} cl_mem_ion_host_ptr;

#endif /* CL_VERSION_1_1 */


#ifdef CL_VERSION_2_0
/*********************************
* cl_khr_sub_groups extension
*********************************/
#define cl_khr_sub_groups 1

//typedef cl_uint  cl_kernel_sub_group_info;

/* cl_khr_sub_group_info */
#define CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR	0x2033
#define CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR		0x2034

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelSubGroupInfoKHR(cl_kernel /* in_kernel */,
						   cl_device_id /*in_device*/,
						   cl_kernel_sub_group_info /* param_name */,
						   size_t /*input_value_size*/,
						   const void * /*input_value*/,
						   size_t /*param_value_size*/,
						   void* /*param_value*/,
						   size_t* /*param_value_size_ret*/ ) CL_EXT_SUFFIX__VERSION_2_0;
						   
typedef CL_API_ENTRY cl_int
     ( CL_API_CALL * clGetKernelSubGroupInfoKHR_fn)(cl_kernel /* in_kernel */,
						      cl_device_id /*in_device*/,
						      cl_kernel_sub_group_info /* param_name */,
						      size_t /*input_value_size*/,
						      const void * /*input_value*/,
						      size_t /*param_value_size*/,
						      void* /*param_value*/,
						      size_t* /*param_value_size_ret*/ ) CL_EXT_SUFFIX__VERSION_2_0;
#endif /* CL_VERSION_2_0 */

#ifdef CL_VERSION_2_1
/*********************************
* cl_khr_priority_hints extension
*********************************/
#define cl_khr_priority_hints 1

typedef cl_uint  cl_queue_priority_khr;

/* cl_command_queue_properties */
#define CL_QUEUE_PRIORITY_KHR 0x1096

/* cl_queue_priority_khr */
#define CL_QUEUE_PRIORITY_HIGH_KHR (1<<0)
#define CL_QUEUE_PRIORITY_MED_KHR (1<<1)
#define CL_QUEUE_PRIORITY_LOW_KHR (1<<2)

#endif /* CL_VERSION_2_1 */

#ifdef CL_VERSION_2_1
/*********************************
* cl_khr_throttle_hints extension
*********************************/
#define cl_khr_throttle_hints 1

typedef cl_uint  cl_queue_throttle_khr;

/* cl_command_queue_properties */
#define CL_QUEUE_THROTTLE_KHR 0x1097

/* cl_queue_throttle_khr */
#define CL_QUEUE_THROTTLE_HIGH_KHR (1<<0)
#define CL_QUEUE_THROTTLE_MED_KHR (1<<1)
#define CL_QUEUE_THROTTLE_LOW_KHR (1<<2)

#endif /* CL_VERSION_2_1 */

/* Altera extensions. */


/*********************************
* cl_altera_mem_banks
*********************************/

/* cl_mem_flags - bitfield */
#define CL_MEM_BANK_AUTO_ALTERA           (0<<16)
#define CL_MEM_BANK_1_ALTERA              (1<<16)
#define CL_MEM_BANK_2_ALTERA              (2<<16)
#define CL_MEM_BANK_3_ALTERA              (3<<16)
#define CL_MEM_BANK_4_ALTERA              (4<<16)
#define CL_MEM_BANK_5_ALTERA              (5<<16)
#define CL_MEM_BANK_6_ALTERA              (6<<16)
#define CL_MEM_BANK_7_ALTERA              (7<<16)


#define CL_MEM_HETEROGENEOUS_ALTERA       (1<<19)



/*********************************
* clGetDeviceInfo extension
*********************************/
#define cl_altera_device_temperature
/* Enum query for clGetDeviceInfo to get the die temperature in Celsius as a cl_int.
 * If the device does not support the query then the result will be 0 */
#define CL_DEVICE_CORE_TEMPERATURE_ALTERA        0x40F3



/*********************************
* CL API object tracking.
*********************************/
#define cl_altera_live_object_tracking

/* Call this to begin tracking CL API objects.  
 * Ideally, do this immediately after getting the platform ID.
 * This takes extra space and time.
 */
extern CL_API_ENTRY void CL_API_CALL
clTrackLiveObjectsIntelFPGA(cl_platform_id platform);

/* Call this to be informed of all the live CL API objects, with their
 * reference counts.
 * The type name argument to the callback will be the string form of the type name
 * e.g. "cl_event" for a cl_event.
 */
extern CL_API_ENTRY void CL_API_CALL
clReportLiveObjectsIntelFPGA(
      cl_platform_id platform,
      void (CL_CALLBACK * /*report_fn*/)(
         void* /* user_data */,
         void* /* obj_ptr */,
         const char* /* type_name */, 
         cl_uint /* refcount */ ),
      void* /* user_data*/ );

/* Call this to query the FPGA and collect dynamic profiling data
 * for a single kernel.
 *
 * The event passed to this call must be the event used
 * in the kernel clEnqueueNDRangeKernel call. If the kernel
 * completes execution before this function is invoked, 
 * this function will return an event error code.
 *
 * NOTE: 
 * Invoking this function while the kernel is running will
 * disable the profile counters for a given interval.
 * For example, on a PCIe-based system this was measured
 * to be approximately 100us.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetProfileInfoIntelFPGA(
      cl_event /* kernel event */
      );

/*********************************
* Altera offline compiler modes, offline device emulation.
*********************************/
#define cl_altera_compiler_mode

#define CL_CONTEXT_COMPILER_MODE_ALTERA 0x40F0

#define CL_CONTEXT_COMPILER_MODE_OFFLINE_ALTERA 0
#define CL_CONTEXT_COMPILER_MODE_OFFLINE_CREATE_EXE_LIBRARY_ALTERA 1
#define CL_CONTEXT_COMPILER_MODE_OFFLINE_USE_EXE_LIBRARY_ALTERA 2
#define CL_CONTEXT_COMPILER_MODE_PRELOADED_BINARY_ONLY_ALTERA 3

/* This property is used to specify the root directory of
 * the executable program library for compiler modes 
 * CL_CONTEXT_COMPILER_MODE_OFFLINE_CREATE_EXE_LIBRARY and
 * CL_CONTEXT_COMPILER_MODE_OFFLINE_USE_EXE_LIBRARY.
 * The value should be a pointer to a C-style character string naming
 * the directory.  It can be relative, but will be resolved to an absolute
 * directory at context creation time.
 */
#define CL_CONTEXT_PROGRAM_EXE_LIBRARY_ROOT_ALTERA 0x40F1

/* This property is used to emulate, as much as possible,
 * having a device that is actually not attached.
 * Kernels may be enqueued but their code will not be run, 
 * so data coming back from the device may be invalid.
 * The value should be a pointer to a C-style character string with the
 * short name for the device.
 */
#define CL_CONTEXT_OFFLINE_DEVICE_ALTERA 0x40F2

/* FCD Support for Board Specific Functions */

extern CL_API_ENTRY void* CL_API_CALL 
clGetBoardExtensionFunctionAddressIntelFPGA(const char * /* func_name */,
                                         cl_device_id    /* device */);

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinaryAndProgramDeviceIntelFPGA(cl_context                     /* context */,
                          cl_uint                        /* num_devices */,
                          const cl_device_id *           /* device_list */,
                          const size_t *                 /* lengths */,
                          const unsigned char **         /* binaries */,
                          cl_int *                       /* binary_status */,
                          cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
clSetCommandQueueProperty(cl_command_queue              /* command_queue */,
                          cl_command_queue_properties   /* properties */, 
                          cl_bool                        /* enable */,
                          cl_command_queue_properties * /* old_properties */) CL_EXT_SUFFIX__VERSION_1_0_DEPRECATED;

/* Our own extra APIs */

extern CL_API_ENTRY cl_int CL_API_CALL
clReconfigurePLLIntelFPGA(
         cl_device_id device,
         const char *pll_settings_str);


extern CL_API_ENTRY cl_int CL_API_CALL
clResetKernelsIntelFPGA(
         cl_context context,
         cl_uint num_devices,
         const cl_device_id *device_list);

#ifdef __cplusplus
}
#endif


#endif /* __CL_EXT_H */
