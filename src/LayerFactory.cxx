// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
#pragma once

#include "Layers\LayerFactory.h"
#include <iostream>

#include <Layers\Flatten.h>
#include <Layers\Dense.h>
#include <Layers\Conv2D.h>
#include <Layers\Conv2DTransposeCL.h>
#include <Layers\Conv2DPCA.h>
#include <Layers\MaxPooling2D.h>
#include <Layers\GlobalAvgPooling2D.h>
#include <Layers\BatchNormalization.h>
#include <Layers\ReLU.h>
#include <Layers\ZeroPadding2D.h>
#include <Layers\Input.h>
#include <Layers\Add.h>
#include <Layers\Concatenate.h>
#include <Layers\DepthToSpace.h>
#include <Layers\DistributedConv2D.h>
#include <Layers\AddCL.h>
#include <Layers\BatchNormalizationCL.h>
#include <Layers\MaxPooling2DCL.h>
#include <Layers\ReLUCL.h>
#include <Layers\ConcatenateCL.h>
#include <OpenCL\clContext.h>
#include <Layers\DenseCL.h>
#include <Layers\Conv2DCL.h>

namespace tf2ocl {

	LayerFactory::LayerFactory()
	{

	}

	LayerFactory::~LayerFactory()
	{

	}

	bool LayerFactory::build_layers(const rapidjson::Value& layer_config, std::map<std::string, Layer*> &layers)
	{
		if (layer_config.Size() == 0)
			return false;

		bool useocl = CLContext::Instance()->hasContext();

#pragma omp parallel for
		for (int i = 0; i < layer_config.Size(); i++)
		{
			Layer* _layer = NULL;

			const rapidjson::Value& layer = layer_config[i];
			const std::string type = layer["type"].GetString();
			const std::string name = layer["name"].GetString();
			
			if (type.compare("Flatten") == 0)
			{
				_layer = new Flatten(layer);
				
			}
			else if (type.compare("Dense") == 0)
			{
				if (useocl)
				{
					DenseCL* l = new DenseCL(layer);
					bool compiled = l->compile();
					if (!compiled)
						throw new std::runtime_error("Failed to compile a layer!");

					_layer = l;
				}
				else
				{
					_layer = new Dense(layer);
				}
			}
			else if (type.compare("Conv2D") == 0)
			{
				if (useocl)
				{
					Conv2DCL* l = new Conv2DCL(layer);
					bool compiled = l->compile();

					if (!compiled) {
						throw new std::runtime_error("Failed to compile a layer!");
					}
					_layer = l;
				}
				else
				{
					_layer = new Conv2D(layer);
				}
			}
			else if (type.compare("MaxPooling2D") ==0)
			{
				if (useocl)
				{
					MaxPooling2DCL* l = new MaxPooling2DCL(layer);
					bool compiled = l->compile();

					if (!compiled) {
						throw new std::runtime_error("Failed to compile a layer!");
					}
					_layer = l;
				}
				else
				{
					_layer = new MaxPooling2D(layer);
				}
			}
			else if (type.compare("GlobalAveragePooling2D") == 0)
			{
				_layer = new GlobalAvgPooling2D(layer);
			}
			else if (type.compare("BatchNormalization") == 0)
			{
				if (useocl)
				{
					BatchNormalizationCL* l = new BatchNormalizationCL(layer);
					bool compiled = l->compile();

					if (!compiled) {
						throw new std::runtime_error("Failed to compile a layer!");
					}
					_layer = l;
				}
				else
				{
					_layer = new BatchNormalization(layer);
				}
			}
			else if (type.compare("ReLU") == 0)
			{
				if (useocl)
				{
					ReLUCL* l = new ReLUCL(layer);
					bool compiled = l->compile();

					if (!compiled) {
						throw new std::runtime_error("Failed to compile a layer!");
					}
					_layer = l;
				}
				else
				{
					_layer = new ReLU(layer);
				}
			}
			else if (type.compare("ZeroPadding2D") == 0)
			{
				_layer = new ZeroPadding2D(layer);
			}
			else if (type.compare("InputLayer") == 0)
			{
				_layer = new Input(layer);
			}
			else if (type.compare("Add") == 0)
			{
				if (useocl)
				{
					AddCL* l = new AddCL(layer);
					bool compiled = l->compile();

					if (!compiled) {
						throw new std::runtime_error("Failed to compile a layer!");
					}
					_layer = l;
				}
				else
				{
					_layer = new Add(layer);
				}
			}
			else if (type.compare("Conv2DTranspose") == 0)
			{
				if (useocl)
				{
					Conv2DTransposeCL* l = new Conv2DTransposeCL(layer);
					bool compiled = l->compile();

					if (!compiled) {
						throw new std::runtime_error("Failed to compile a layer!");
					}
					_layer = l;
				}
				else
				{
					_layer = new Conv2DTranspose(layer);
				}
			}
			else if (type.compare("Concatenate") == 0)
			{
				if (useocl)
				{
					ConcatenateCL* l = new ConcatenateCL(layer);
					bool compiled = l->compile();

					if (!compiled) {
						throw new std::runtime_error("Failed to compile a layer!");
					}
					_layer = l;
				}
				else
				{
					_layer = new Concatenate(layer);
				}
			}
			else if (type.compare("DepthToSpace") == 0)
			{
				_layer = new DepthToSpace(layer);
			}
			else
			{
				throw std::runtime_error("Un-supported Layer encountered!");
			}

#pragma omp critical
			{
				layers.insert(std::pair<std::string, Layer*>(name, _layer));
			}
		}

		return true;
	}	
}