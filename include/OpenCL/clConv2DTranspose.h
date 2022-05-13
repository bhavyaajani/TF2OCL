// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "clOperation.h"

namespace tf2ocl {
	class clConv2DTranspose :public clOperator {

	public:
		clConv2DTranspose();
		virtual ~clConv2DTranspose();

		bool call(cl_mem d_x, cl_mem d_k, cl_mem d_b, cl_mem d_y, unsigned int iH, unsigned int iW, unsigned int D, 
			unsigned int oH, unsigned int oW, unsigned int C, unsigned int relu);

	protected:
		virtual std::string getKernelSourceFileName(std::string ocl_vendor);

		virtual std::string getKernelBinaryFileName(std::string ocl_vendor);
		
		virtual bool _compile(CLContext *context);

		size_t _workgroup_size;
		size_t _workgroup_mem;
	};
}