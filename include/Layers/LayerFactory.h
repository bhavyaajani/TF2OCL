// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "rapidjson/document.h"
#include "Layers\Layer.h"

#include <map>

namespace tf2ocl {

	typedef std::map<std::string, Layer *> LayerContainerType; 

	class LayerFactory {

	public:
		 
		static bool build_layers(const rapidjson::Value& layer_config, LayerContainerType &layers);

	protected:
		LayerFactory();

		virtual ~LayerFactory();
	};
}