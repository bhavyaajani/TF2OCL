// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clAdd.h"

namespace tf2ocl {
	
	clAdd::clAdd()
	{
	}

	clAdd::~clAdd()
	{
	}

	std::string clAdd::getKernelSourceFileName(std::string ocl_vendor)
	{
		return "clAdd.cl";
	}

	std::string clAdd::getKernelBinaryFileName(std::string ocl_vendor)
	{
		throw new std::runtime_error("Not implemented yet !");
	}

	bool clAdd::call(cl_mem d_a, cl_mem d_b, cl_mem d_c, unsigned int count)
	{
		if (!_compiled)
			return false;

		bool status = false;

		try {
			//Important to set this otherwise cause weird OpenCL errors!
			cl_int err = CL_SUCCESS;

			size_t globalSize = count;
			size_t localSize = 16;

			// Set the arguments to our compute kernel
			err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
			err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);

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