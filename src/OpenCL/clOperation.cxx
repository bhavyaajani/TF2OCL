// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include "OpenCL\clOperation.h"

namespace tf2ocl {

	clOperator::clOperator()
	{
		_compiled = false;
		_clsource = NULL;
	}

	clOperator::~clOperator()
	{
		release();
	}

	void clOperator::release()
	{
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(queue);
	}

	bool clOperator::loadSourceFromFile(std::string filename)
	{
		if (_clsource)
		{
			free(_clsource);
			_clsource = NULL;
		}

		FILE *fp;

		size_t source_size, program_size;

		fp = fopen(filename.c_str(), "rb");
		if (!fp) {
			return false;
		}

		fseek(fp, 0, SEEK_END);
		program_size = ftell(fp);
		rewind(fp);
		_clsource = (char*)malloc(program_size + 1);
		_clsource[program_size] = '\0';
		fread(_clsource, sizeof(char), program_size, fp);
		fclose(fp);

		//std::cout << _clsource << std::endl;
		return true;
	}

	// Loads a file in binary form.
	unsigned char *clOperator::loadBinaryFile(const char *file_name, size_t *size) {
		// Open the File
		FILE* fp;
		long ftell_size;
		size_t elements_read;
#ifdef _WIN32
		if (fopen_s(&fp, file_name, "rb") != 0) {
			return NULL;
		}
#else
		fp = fopen(file_name, "rb");
		if (fp == 0) {
			return NULL;
		}
#endif

		// Get the size of the file
		fseek(fp, 0, SEEK_END);
		ftell_size = ftell(fp);
		if (ftell_size < 0) {
			fclose(fp);
			return NULL;
		}
		*size = (unsigned)ftell_size;

		// Allocate space for the binary
		unsigned char *binary = new unsigned char[*size];

		// Go back to the file start
		rewind(fp);

		// Read the file into the binary
		elements_read = fread((void*)binary, *size, 1, fp);
		if (elements_read == 0) {
			delete[] binary;
			fclose(fp);
			return NULL;
		}

		fclose(fp);
		return binary;
	}

	bool clOperator::createProgramWithSource(CLContext *context, cl_int &err)
	{
		std::string filePath = getKernelSourceFileName(context->getVendorString());

		if (filePath.empty())
			return false;

		bool load = loadSourceFromFile(filePath);
		if (!load)
		{
			return false;
		}

		// Create the compute program from the source buffer
		program = clCreateProgramWithSource(context->getContext(), 1, (const char **)& _clsource, NULL, &err);

		return (err == CL_SUCCESS);
	}

	bool clOperator::createProgramWithBinary(CLContext *context, cl_int &err)
	{
		std::string binary_file = getKernelBinaryFileName(context->getVendorString());

		if (binary_file.empty())
		{
			return false;
		}

		size_t binary_size;
		unsigned char* binary = loadBinaryFile(binary_file.c_str(), &binary_size);

		// Create the compute program from the pre-compiled binary
		cl_device_id deviceId = context->getDeviceId();
		program = clCreateProgramWithBinary(context->getContext(), 1, &deviceId, &binary_size,
			(const unsigned char**)&binary, NULL, &err);

		//TODO: Free binary file in memory!
		return (err == CL_SUCCESS);
	}

	bool clOperator::compile()
	{
		CLContext *context = CLContext::Instance();
		if (context)
		{
			return this->_compile(context);
		}
		else
		{
			throw new std::runtime_error("No valid CLContext found! ");
		}
	}

	bool clOperator::_compile(CLContext *context)
	{
		if (_compiled)
			return false;

		_compiled = false;

		try {

			cl_int err;
			cl_device_id deviceId = context->getDeviceId();

			cl_context oclcontext = context->getContext();

			std::string ocl_vendor = context->getVendorString();
			//Create Program
			if (ocl_vendor.find("Arria") != std::string::npos)
			{
				_compiled = createProgramWithBinary(context, err);
			}
			else
			{
				_compiled = createProgramWithSource(context, err);
			}

			if (!_compiled)
			{
				throw std::runtime_error("Failed to build OpenCL program !");
			}

			// Create a command queue
			queue = clCreateCommandQueue(oclcontext, deviceId, 0, &err);
			CHECK_AND_THROW(err);

			// Build the program executable
			err |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
			CHECK_AND_THROW(err);
			// Create the compute kernel in the program we wish to run
			kernel = clCreateKernel(program, "call", &err);
			CHECK_AND_THROW(err);

			_compiled = true;
		}
		catch (std::runtime_error &e) {
			release();
		}

		return _compiled;
	}

	cl_context clOperator::_context()
	{
		CLContext *context = CLContext::Instance();
		if (context)
		{
			return context->getContext();
		}

		return 0;
	}

	cl_device_id clOperator::_device()
	{
		CLContext *context = CLContext::Instance();
		if (context)
		{
			return context->getDeviceId();
		}

		return 0;
	}
}