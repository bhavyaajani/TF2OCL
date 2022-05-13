// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "clOperation.h"

namespace tf2ocl {
	class clMatVecMul : public clOperator {

	public:
		clMatVecMul();
		virtual ~clMatVecMul();

		bool call(float* M, float* X, float* Y, float* B, unsigned int H, unsigned int W);
		
	protected:
		virtual std::string getKernelSourceFileName(std::string ocl_vendor);

		virtual std::string getKernelBinaryFileName(std::string ocl_vendor);
	};
}