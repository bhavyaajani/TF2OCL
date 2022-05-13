// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "clOperation.h"

namespace tf2ocl {
	class clConcate :public clOperator {

	public:
		clConcate();
		virtual ~clConcate();

		bool call(cl_mem d_a, cl_mem d_b, cl_mem d_c, unsigned int H, unsigned int W, unsigned int C1,
			unsigned int C2);

	protected:
		virtual std::string getKernelSourceFileName(std::string ocl_vendor);

		virtual std::string getKernelBinaryFileName(std::string ocl_vendor);
		
	};
}