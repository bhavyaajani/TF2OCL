// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "rapidjson/document.h"
#include "tensor.h"

namespace tf2ocl {

	class Layer {

	public:
		Layer(const rapidjson::Value &value);

		virtual ~Layer();

		virtual Tensor* call(std::vector<Tensor*> inputs);

		std::string name()
		{
			return this->_name;
		}

		std::string type()
		{
			return this->_type;
		}

		void setDirty()
		{
			this->_dirty = true;
		}
		bool isDirty()
		{
			return _dirty;
		}

	protected:
		virtual bool operator()(std::vector<Tensor*> inputs) = 0;

		void check_inputs(std::vector<Tensor*> inputs);

		virtual void build_outputs();

		virtual TensorMemType layerInputTensorMemType()
		{
			return TensorMemType::HOST;
		}

		virtual TensorMemType layerOutputTensorMemType()
		{
			return TensorMemType::HOST;
		}

		std::string _name;
		std::string _type;
		TensorNDim _inputNDim;
		TensorNDim _outputNDim;
		Tensor* _output;

		bool _dirty;
	};
}