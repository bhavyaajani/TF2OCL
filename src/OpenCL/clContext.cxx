// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clContext.h"

namespace tf2ocl {

	CLContext* CLContext::_instance = NULL;

	CLContext::CLContext()
	{
		_context = 0;
		_deviceId = 0;
		_ocl_vendor = "";
	}

	CLContext::~CLContext()
	{
		if (_context)
			clReleaseContext(_context);
	}

	bool CLContext::bind(std::string ocl_vendor)
	{
		if (_context == 0)
		{
			if (ocl_vendor.empty())
				return false;

			//clPrintAvailableDevInfo(CL_DEVICE_TYPE_ACCELERATOR);

			cl_int err;

			cl_device_id deviceId;
			bool found = clGetDeviceIdByVendorName(ocl_vendor.c_str(), &deviceId);
			if (!found)
			{
				throw std::runtime_error("No valid device found!");
			}

			cl_context context = clCreateContext(0, 1, &deviceId, NULL, NULL, &err);
			if (err != CL_SUCCESS)
			{
				throw std::runtime_error("Failed to get valid OCL context!");
			}

			_context = context;
			_deviceId = deviceId;
			return true;
		}

		return false;
	}

	void CLContext::clPrintDevInfo(cl_device_id device) {
		char device_string[1024];

		// CL_DEVICE_NAME
		clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
		printf("  CL_DEVICE_NAME: \t\t\t%s\n", device_string);

		// CL_DEVICE_VENDOR
		clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
		printf("  CL_DEVICE_VENDOR: \t\t\t%s\n", device_string);

		// CL_DRIVER_VERSION
		clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
		printf("  CL_DRIVER_VERSION: \t\t\t%s\n", device_string);

		// CL_DEVICE_INFO
		cl_device_type type;
		clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
		if (type & CL_DEVICE_TYPE_CPU)
			printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
		if (type & CL_DEVICE_TYPE_GPU)
			printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
		if (type & CL_DEVICE_TYPE_ACCELERATOR)
			printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
		if (type & CL_DEVICE_TYPE_DEFAULT)
			printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");

		// CL_DEVICE_MAX_COMPUTE_UNITS
		cl_uint compute_units;
		clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
		printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);

		// CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
		size_t workitem_dims;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
		printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", workitem_dims);

		// CL_DEVICE_MAX_WORK_ITEM_SIZES
		size_t workitem_size[3];
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
		printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", workitem_size[0], workitem_size[1], workitem_size[2]);

		// CL_DEVICE_MAX_WORK_GROUP_SIZE
		size_t workgroup_size;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
		printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u\n", workgroup_size);

		// CL_DEVICE_MAX_CLOCK_FREQUENCY
		cl_uint clock_frequency;
		clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
		printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz\n", clock_frequency);

		// CL_DEVICE_ADDRESS_BITS
		cl_uint addr_bits;
		clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
		printf("  CL_DEVICE_ADDRESS_BITS:\t\t%u\n", addr_bits);

		// CL_DEVICE_MAX_MEM_ALLOC_SIZE
		cl_ulong max_mem_alloc_size;
		clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
		printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

		// CL_DEVICE_GLOBAL_MEM_SIZE
		cl_ulong mem_size;
		clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
		printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));

		// CL_DEVICE_ERROR_CORRECTION_SUPPORT
		cl_bool error_correction_support;
		clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
		printf("  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", error_correction_support == CL_TRUE ? "yes" : "no");

		// CL_DEVICE_LOCAL_MEM_TYPE
		cl_device_local_mem_type local_mem_type;
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
		printf("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", local_mem_type == 1 ? "local" : "global");

		// CL_DEVICE_LOCAL_MEM_SIZE
		clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
		printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));

		// CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
		clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
		printf("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(mem_size / 1024));

		// CL_DEVICE_QUEUE_PROPERTIES
		cl_command_queue_properties queue_properties;
		clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
		if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
			printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
		if (queue_properties & CL_QUEUE_PROFILING_ENABLE)
			printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

		// CL_DEVICE_IMAGE_SUPPORT
		cl_bool image_support;
		clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
		printf("  CL_DEVICE_IMAGE_SUPPORT:\t\t%u\n", image_support);

		// CL_DEVICE_MAX_READ_IMAGE_ARGS
		cl_uint max_read_image_args;
		clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
		printf("  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u\n", max_read_image_args);

		// CL_DEVICE_MAX_WRITE_IMAGE_ARGS
		cl_uint max_write_image_args;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
		printf("  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", max_write_image_args);

		// CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
		size_t szMaxDims[5];
		printf("\n  CL_DEVICE_IMAGE <dim>");
		clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
		printf("\t\t\t2D_MAX_WIDTH\t %u\n", szMaxDims[0]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
		printf("\t\t\t\t\t2D_MAX_HEIGHT\t %u\n", szMaxDims[1]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
		printf("\t\t\t\t\t3D_MAX_WIDTH\t %u\n", szMaxDims[2]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
		printf("\t\t\t\t\t3D_MAX_HEIGHT\t %u\n", szMaxDims[3]);
		clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
		printf("\t\t\t\t\t3D_MAX_DEPTH\t %u\n", szMaxDims[4]);

		// CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
		printf("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
		cl_uint vec_width[6];
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
		clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
		printf("CHAR %u, SHORT %u, INT %u, FLOAT %u, DOUBLE %u\n\n\n",
			vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4]);
	}

	bool CLContext::clGetDeviceIdByVendorName(const char* vendor_name, cl_device_id *out)
	{
		cl_uint platformCount;
		cl_platform_id* platforms;

		bool found = false;

		// get all platforms
		clGetPlatformIDs(0, NULL, &platformCount);
		platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
		clGetPlatformIDs(platformCount, platforms, NULL);

		for (int i = 0; (!found) & (i < platformCount); i++) {

			cl_uint deviceCount;
			cl_device_id* devices;
			// get all devices
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
			devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
			clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

			for (int j = 0; (!found) & (j < deviceCount); j++)
			{
				char* value;
				size_t valueSize;
				clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
				value = (char*)malloc(valueSize);
				clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);

				if (std::strstr(value, vendor_name))
				{
					*out = devices[j];
					found = true;
				}

				free(value);
			}

			free(devices);

		}

		free(platforms);
		return found;
	}

	void CLContext::clPrintAvailableDevInfo(int deviceType)
	{
		cl_uint platformCount;
		cl_platform_id* platforms;

		// get all platforms
		clGetPlatformIDs(0, NULL, &platformCount);
		platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
		clGetPlatformIDs(platformCount, platforms, NULL);

		for (int i = 0; i < platformCount; i++) {
			printf("Platform: %d\n", i);

			cl_uint deviceCount;
			cl_device_id* devices;
			// get all devices
			clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &deviceCount);
			devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
			clGetDeviceIDs(platforms[i], deviceType, deviceCount, devices, NULL);

			// for each device print critical attributes
			for (int j = 0; j < deviceCount; j++)
			{
				clPrintDevInfo(devices[j]);
			}

			free(devices);
		}

		free(platforms);
	}
}