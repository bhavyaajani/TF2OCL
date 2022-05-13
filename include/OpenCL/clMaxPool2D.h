// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "clOperation.h"

namespace tf2ocl {
	class clMaxPool2D :public clOperator {

	public:
		clMaxPool2D();
		virtual ~clMaxPool2D();

		bool call(cl_mem d_I, cl_mem d_O, unsigned int iH, unsigned int iW, unsigned int D, 
			unsigned int oH, unsigned int oW, unsigned int pH, unsigned int pW, unsigned int sH, unsigned int sW);

	protected:
		virtual std::string getKernelSourceFileName(std::string ocl_vendor);

		virtual std::string getKernelBinaryFileName(std::string ocl_vendor);
		
	};
}