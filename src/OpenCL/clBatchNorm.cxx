// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clBatchNorm.h"

namespace tf2ocl {
	
	clBatchNorm::clBatchNorm()
	{

	}

	clBatchNorm::~clBatchNorm()
	{

	}

	std::string clBatchNorm::getKernelSourceFileName(std::string ocl_vendor)
	{
		return "clBatchNorm.cl";
	}

	std::string clBatchNorm::getKernelBinaryFileName(std::string ocl_vendor)
	{
		throw new std::runtime_error("Not implemented yet !");
	}

	bool clBatchNorm::call(cl_mem d_I, cl_mem d_O, cl_mem _mean, cl_mem _variance, cl_mem _gamma, cl_mem _beta,
		unsigned int D, unsigned int count)
	{
		if (!_compiled)
			return false;

		bool status = false;

		try {
			//Important to set this otherwise causing OpenCL errors!
			cl_int err = CL_SUCCESS;

			size_t globalSize = count;
			size_t localSize = 32;//Don't care!

			// Set the arguments to our compute kernel
			err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_I);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_O);
			err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &_mean);
			err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &_variance);
			err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &_gamma);
			err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &_beta);
			err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &D);
			//err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &count);
			
			// Execute the kernel over the entire range of the data set 
			err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
				0, NULL, NULL);

			// Wait for the command queue to get serviced before reading back results
			clFinish(queue);
			CHECK_AND_THROW(err);

			status = true;
		}
		catch (std::runtime_error &e) {
			
		}
		
		return status;
	}
}