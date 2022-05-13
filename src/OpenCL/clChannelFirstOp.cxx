// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clChannelFirstOp.h"

namespace tf2ocl {
	
	clChannelFirstOp::clChannelFirstOp()
	{
	}

	clChannelFirstOp::~clChannelFirstOp()
	{
	}

	std::string clChannelFirstOp::getKernelSourceFileName(std::string ocl_vendor)
	{
		return "clChannelFirstOp.cl";
	}

	std::string clChannelFirstOp::getKernelBinaryFileName(std::string ocl_vendor)
	{
		return "";
	}

	bool clChannelFirstOp::call(float* Kin, float* KOut, unsigned int H, unsigned int W, unsigned int D, unsigned int C)
	{
		if (!_compiled)
			return false;

		bool status = false;

		// Device input buffers
		cl_mem d_kin;
		cl_mem d_kout;
		
		try {
			cl_int err;

			size_t K_bytes = H*W*D*C*sizeof(float);

			// Create the input and output arrays in device memory for our calculation
			d_kin = clCreateBuffer(_context(), CL_MEM_READ_ONLY, K_bytes, NULL, &err);
			d_kout = clCreateBuffer(_context(), CL_MEM_WRITE_ONLY, K_bytes, NULL, &err);
			CHECK_AND_THROW(err);

			// Write our data set into the input array in device memory
			err |= clEnqueueWriteBuffer(queue, d_kin, CL_TRUE, 0,
				K_bytes, Kin, 0, NULL, NULL);

			CHECK_AND_THROW(err);

			size_t globalSize = H*W*D*C;
			size_t localSize = 1;
			
			// Set the arguments to our compute kernel
			err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_kin);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_kout);
			err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &H);
			err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &W);
			err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &D);
			err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &C);

			CHECK_AND_THROW(err);

			// Execute the kernel over the entire range of the data set 
			err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
				0, NULL, NULL);

			// Wait for the command queue to get serviced before reading back results
			clFinish(queue);
			CHECK_AND_THROW(err);
			// Read the results from the device
			err |= clEnqueueReadBuffer(queue, d_kout, CL_TRUE, 0,
				K_bytes, KOut, 0, NULL, NULL);
			CHECK_AND_THROW(err);

			status = true;
		}
		catch (std::runtime_error &e) {
			
		}
		clReleaseMemObject(d_kin);
		clReleaseMemObject(d_kout);
		
		return status;
	}
}