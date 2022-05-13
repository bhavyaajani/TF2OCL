// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\BatchNormalizationCL.h"
#include "OpenCL\clBatchNorm.h"
#include <iostream>

namespace tf2ocl {

	BatchNormalizationCL::BatchNormalizationCL(const rapidjson::Value &config) : BatchNormalization(config), LayerCL()
	{
		_operator = new clBatchNorm();
	}

	BatchNormalizationCL::~BatchNormalizationCL()
	{
		if (_operator)
			delete _operator;
	}

	bool BatchNormalizationCL::compile()
	{
		if (!_compiled) {
			_compiled = _operator->compile();

			if (_compiled)
			{
				_compiled &= this->_mean->allocateDevice(CL_MEM_READ_ONLY);
				_compiled &= this->_mean->memcopyHostToDevice();

				_compiled &= this->_variance->allocateDevice(CL_MEM_READ_ONLY);
				_compiled &= this->_variance->memcopyHostToDevice();

				if (this->_gamma)
				{
					_compiled &= this->_gamma->allocateDevice(CL_MEM_READ_ONLY);
					_compiled &= this->_gamma->memcopyHostToDevice();
				}

				if (this->_beta)
				{
					_compiled &= this->_beta->allocateDevice(CL_MEM_READ_ONLY);
					_compiled &= this->_beta->memcopyHostToDevice();
				}
			}
		}

		return _compiled;
	}

	bool BatchNormalizationCL::operator()(std::vector<Tensor*> inputs)
	{
		if (inputs.size() != 1)
		{
			throw std::runtime_error("Expect 1 inputs!");
		}

		unsigned int count = _output->count();
		unsigned int D = _mean->count();//D

		_operator->call(inputs[0]->device(),_output->device(),_mean->device(),_variance->device(),
			_gamma->device(),_beta->device(),D,count);

		return true;
	}
}