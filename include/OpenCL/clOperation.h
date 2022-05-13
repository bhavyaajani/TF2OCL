// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include <OpenCL\clContext.h>
#include <stdexcept>
#include <iostream>

namespace tf2ocl {

	class clOperator {
	public:

		clOperator();
		virtual ~clOperator();

		bool compile();

		inline bool isCompiled()
		{
			return _compiled;
		}

	protected:
		virtual bool _compile(CLContext *context);

		virtual std::string getKernelSourceFileName(std::string ocl_vendor) = 0;

		virtual std::string getKernelBinaryFileName(std::string ocl_vendor) = 0;

		bool createProgramWithBinary(CLContext *context, cl_int &err);

		bool createProgramWithSource(CLContext *context, cl_int &err);

		bool loadSourceFromFile(std::string filename);

		unsigned char *loadBinaryFile(const char *file_name, size_t *size);

		inline void CHECK_AND_THROW(cl_int err)
		{
			if (err != CL_SUCCESS)
			{
				std::cout << "OpenCL Error: " << err << std::endl;
				throw std::runtime_error("OpenCL Error!");
			}
		}

		virtual void release();

		cl_context _context();

		cl_device_id _device();

		cl_command_queue queue;           // command queue
		cl_program program;               // program
		cl_kernel kernel;                 // kernel
		bool _compiled;
		char* _clsource;
	};
}