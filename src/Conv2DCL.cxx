// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\Conv2DCL.h"
#include "OpenCL\clConv2D.h"
#include "OpenCL\clChannelFirstOp.h"

#include <iostream>

namespace tf2ocl {

	Conv2DCL::Conv2DCL(const rapidjson::Value &config) : Conv2D(config), LayerCL()
	{
		_kernelCHWD = NULL;
		_operator = new clConv2D();
	}

	Conv2DCL::~Conv2DCL()
	{
		if (_operator)
			delete _operator;

		if (_kernelCHWD)
			delete _kernelCHWD;
	}	

	bool Conv2DCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();
		}

		/**
		if (_compiled) {
			clChannelFirstOp ops;
			_compiled &= ops.compile(deviceId, ocl_vendor);

			if (false)
			{
				float *Kin = (float*) this->_kernel->host();
				unsigned int H = _kernel->ndim()[0];
				unsigned int W = _kernel->ndim()[1];
				unsigned int D = _kernel->ndim()[2];
				unsigned int C = _kernel->ndim()[3];

				TensorNDim ndim; ndim.push_back(C); ndim.push_back(H); ndim.push_back(W); ndim.push_back(D);

				_kernelCHWD = new Tensor(ndim, TensorDType::Float);
				_kernelCHWD->allocate();

				float* KOut = (float*) _kernelCHWD->host();

				bool status = ops.call(Kin, KOut, H, W, D, C);
				if (!status)
				{
					delete _kernelCHWD;
					_kernelCHWD = NULL;
					_compiled = false;
				}
			}
		}
		*/
		if (_compiled)
		{
			_compiled &= this->_kernel->allocateDevice(CL_MEM_READ_ONLY);
			_compiled &= this->_kernel->memcopyHostToDevice();

			if (this->_bias)
			{
				_compiled &= this->_bias->allocateDevice(CL_MEM_READ_ONLY);
				_compiled &= this->_bias->memcopyHostToDevice();
			}
		}
		return _compiled;
	}

	bool Conv2DCL::operator()(std::vector<Tensor*> inputs)
	{
		Tensor *input = inputs[0];

		if (this->_inputNDim != input->ndim())
		{
			throw std::runtime_error("Input Tensor NDim does not match required!");
		}

		if (!_compiled)
		{
			throw std::runtime_error("OpenCL layer not compiled yet !");
		}
		
		unsigned int iH = input->ndim()[0];
		unsigned int iW = input->ndim()[1];
		unsigned int D = input->ndim()[2];

		unsigned int oH = _output->ndim()[0];
		unsigned int oW = _output->ndim()[1];
		unsigned int C = _output->ndim()[2];

		unsigned int kH = this->_kernel->ndim()[0];
		unsigned int kW = this->_kernel->ndim()[1];

		unsigned int sH = _strides[0];
		unsigned int sW = _strides[1];

		unsigned int applyReLU = (_activation.compare("relu") == 0);
		unsigned int applySigmoid = (_activation.compare("sigmoid") == 0);
		unsigned int applyPadding = (_padding.compare("same") == 0);

		_operator->call(input->device(),_kernel->device(), _bias->device(), _output->device(),
			iH, iW, D, kH, kW, oH, oW, C, sH, sW, applyPadding, applyReLU);

		if (applySigmoid)
		{
			_output->allocateHost();
			_output->memcopyDeviceToHost();

			float *Y = (float *)_output->host();
			const unsigned int count = oH*oW*C;

#pragma omp parallel for
			for (int index = 0; index < count; index++)
			{
				Y[index] = 1.0 / (1.0 + std::exp(-Y[index]));
			}

			_output->setDefaultMemType(TensorMemType::HOST);
		}

		return true;
	}
}