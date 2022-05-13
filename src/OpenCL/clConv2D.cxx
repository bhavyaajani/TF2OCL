// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clConv2D.h"

namespace tf2ocl {
	clConv2D::clConv2D()
	{		
	}

	clConv2D::~clConv2D()
	{

	}

	std::string clConv2D::getKernelSourceFileName(std::string ocl_vendor)
	{
		return "clConv2D.cl";
	}

	std::string clConv2D::getKernelBinaryFileName(std::string ocl_vendor)
	{
		return "";
	}

	bool clConv2D::_compile(CLContext *context)
	{
		bool status = clOperator::_compile(context);
		if (status)
		{
			cl_int err = clGetDeviceInfo(context->getDeviceId(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_workgroup_size), &_workgroup_size, NULL);
			CHECK_AND_THROW(err);
		}

		if (status)
		{
			cl_int err = clGetDeviceInfo(context->getDeviceId(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &_workgroup_mem, NULL);
			CHECK_AND_THROW(err);
		}

		return status;
	}

	bool clConv2D::call(cl_mem d_x, cl_mem d_k, cl_mem d_b, cl_mem d_y, unsigned int iH, unsigned int iW, unsigned int D,
		unsigned int kH, unsigned int kW, unsigned int oH, unsigned int oW, unsigned int C, unsigned int sH,
		unsigned int sW, unsigned int padding, unsigned int relu)
	{
		if (!_compiled)
			return false;

		if (kH%2 == 0 || kW%2 == 0)
		{
			std::cout << "Conv2D only with odd kernel dimensions supported !" << std::endl;
			return false;
		}

		//Don't change this without first updating the kernel!
		unsigned int ROLLFACTOR = 1;

		const unsigned int data_block_count = kH*kW*D*ROLLFACTOR;

		const unsigned int data_block_memory = sizeof(cl_float) * data_block_count;
		if (data_block_memory > _workgroup_mem)
		{
			std::cout << "Device Max local memory exceeded !" << std::endl;
			return false;
		}
		
		bool status = false;

		try {
			//Important to set this otherwise causing OpenCL errors!
			cl_int err = CL_SUCCESS;

			// Set the arguments to our compute kernel
			err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_k);
			err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
			err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_y);

			err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &iH);
			err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &iW);
			err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &D);
			err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &kH);
			err |= clSetKernelArg(kernel, 8, sizeof(unsigned int), &kW);
			err |= clSetKernelArg(kernel, 9, sizeof(unsigned int), &oH);
			err |= clSetKernelArg(kernel, 10, sizeof(unsigned int), &oW);
			err |= clSetKernelArg(kernel, 11, sizeof(unsigned int), &C);
			err |= clSetKernelArg(kernel, 12, sizeof(unsigned int), &sH);
			err |= clSetKernelArg(kernel, 13, sizeof(unsigned int), &sW);
			err |= clSetKernelArg(kernel, 14, sizeof(unsigned int), &padding);
			err |= clSetKernelArg(kernel, 15, sizeof(unsigned int), &relu);
			
			CHECK_AND_THROW(err);

			int running_channel_count = 0;
			
			unsigned int RoH = (oH%ROLLFACTOR == 0) ? oH : oH + (ROLLFACTOR - oH%ROLLFACTOR);

			while (running_channel_count < C)
			{
				unsigned int channels_remaining = C - running_channel_count;
				unsigned int channels_current_run = (channels_remaining < _workgroup_size) ? channels_remaining : _workgroup_size;

				unsigned int channel_offset = running_channel_count;

				size_t globalSize = (RoH/ROLLFACTOR)*oW*channels_current_run;
				size_t localSize = channels_current_run;

				size_t reminder = (globalSize%localSize == 0) ? 0 : localSize - (globalSize%localSize);
				globalSize += reminder;
								
				size_t local_memory_size = data_block_memory;

				err |= clSetKernelArg(kernel, 16, local_memory_size, NULL);
				err |= clSetKernelArg(kernel, 17, sizeof(unsigned int), &channel_offset);
				err |= clSetKernelArg(kernel, 18, sizeof(unsigned int), &channels_current_run);
				
				CHECK_AND_THROW(err);

				// Execute the kernel over the entire range of the data set 
				err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
					0, NULL, NULL);

				CHECK_AND_THROW(err);

				running_channel_count += channels_current_run;
			}

			clFinish(queue);
			status = true;
		}
		catch (std::runtime_error &e) {
		
		}

		return status;
	}
}