// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clMatVecMul.h"

namespace tf2ocl {
	
	clMatVecMul::clMatVecMul()
	{
	}

	clMatVecMul::~clMatVecMul()
	{
	}

	std::string clMatVecMul::getKernelSourceFileName(std::string ocl_vendor)
	{
		return "clMatVecMul.cl";
	}

	std::string clMatVecMul::getKernelBinaryFileName(std::string ocl_vendor)
	{
		return "";
	}

	bool clMatVecMul::call(float* M, float* X, float* Y, float* B, unsigned int H, unsigned int W)
	{
		if (!_compiled)
			return false;

		bool status = false;

		// Device input buffers
		cl_mem d_a;
		cl_mem d_b;
		// Device output buffer
		cl_mem d_c;
		cl_mem d_d;

		try {
			cl_int err;

			size_t M_bytes = W*H*sizeof(float);
			size_t X_bytes = W*sizeof(float);
			size_t Y_bytes = H*sizeof(float);
			size_t B_bytes = H*sizeof(float);

			// Create the input and output arrays in device memory for our calculation
			d_a = clCreateBuffer(_context(), CL_MEM_READ_ONLY, M_bytes, NULL, &err);
			d_b = clCreateBuffer(_context(), CL_MEM_READ_ONLY, X_bytes, NULL, &err);
			d_c = clCreateBuffer(_context(), CL_MEM_WRITE_ONLY, Y_bytes, NULL, &err);
			d_d = clCreateBuffer(_context(), CL_MEM_READ_ONLY, B_bytes, NULL, &err);
			CHECK_AND_THROW(err);

			// Write our data set into the input array in device memory
			err |= clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
				M_bytes, M, 0, NULL, NULL);
			err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
				X_bytes, X, 0, NULL, NULL);

			if (B)
			{
				err |= clEnqueueWriteBuffer(queue, d_d, CL_TRUE, 0,
					B_bytes, B, 0, NULL, NULL);
			}

			CHECK_AND_THROW(err);

			size_t globalSize[2], localSize[2];
			size_t threadsPerRow = 8;

			// Number of work items in each local work group
			localSize[0] = 1;
			localSize[1] = threadsPerRow;
			size_t localcount = localSize[0] * localSize[1];
			// Number of total work items - localSize must be devisor
			globalSize[0] = H;
			globalSize[1] = threadsPerRow;

			// Set the arguments to our compute kernel
			err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
			err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
			if (B)
			{
				err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_d);
			}
			else
			{
				cl_mem dummy = 0;
				err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &dummy);
			}
			err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &H);
			err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &W);
			err |= clSetKernelArg(kernel, 6, localcount * sizeof(cl_float), NULL);
			CHECK_AND_THROW(err);

			// Execute the kernel over the entire range of the data set 
			err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize,
				0, NULL, NULL);

			// Wait for the command queue to get serviced before reading back results
			clFinish(queue);
			CHECK_AND_THROW(err);
			// Read the results from the device
			err |= clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
				Y_bytes, Y, 0, NULL, NULL);
			CHECK_AND_THROW(err);

			status = true;
		}
		catch (std::runtime_error &e) {
			
		}
		clReleaseMemObject(d_a);
		clReleaseMemObject(d_b);
		clReleaseMemObject(d_c);
		clReleaseMemObject(d_d);

		return status;
	}
}