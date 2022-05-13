// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clMaxPool2D.h"

namespace tf2ocl {
	
	clMaxPool2D::clMaxPool2D()
	{
	}

	clMaxPool2D::~clMaxPool2D()
	{
	}

	std::string clMaxPool2D::getKernelSourceFileName(std::string ocl_vendor)
	{
		return "clMaxPool2D.cl";
	}

	std::string clMaxPool2D::getKernelBinaryFileName(std::string ocl_vendor)
	{
		throw new std::runtime_error("Not implemented yet !");
	}

	bool clMaxPool2D::call(cl_mem d_I, cl_mem d_O, unsigned int iH, unsigned int iW, unsigned int D, unsigned int oH, 
		unsigned int oW, unsigned int pH, unsigned int pW, unsigned int sH, unsigned int sW)
	{
		if (!_compiled)
			return false;

		bool status = false;

		try {
			//Important to set this otherwise cause weird OpenCL error!
			cl_int err = CL_SUCCESS;
			
			size_t globalSize = oH*oW*D;
			size_t localSize = 16;

			// Set the arguments to our compute kernel
			err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_I);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_O);
			err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &iH);
			err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &iW);
			err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &D);
			err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &oH);
			err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &oW);
			err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &pH);
			err |= clSetKernelArg(kernel, 8, sizeof(unsigned int), &pW);
			err |= clSetKernelArg(kernel, 9, sizeof(unsigned int), &sH);
			err |= clSetKernelArg(kernel, 10, sizeof(unsigned int), &sW);

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