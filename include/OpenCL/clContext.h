// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#pragma once

#include <CL\cl.h>
#include <iostream>

namespace tf2ocl {

	class CLContext {

	public:
		static CLContext* Instance()
		{
			if (!_instance)
			{
				_instance = new CLContext();
			}

			return _instance;
		}

		bool bind(std::string ocl_vendor);

		bool hasContext()
		{
			return (_context != 0);
		}

		cl_context getContext()
		{
			if (_context)
				return _context;
			else
				throw new std::runtime_error("CLContext not binded yet !");
		}

		cl_device_id getDeviceId()
		{
			if (_deviceId)
				return _deviceId;
			else
				throw new std::runtime_error("CLContext not binded yet !");
		}

		std::string getVendorString()
		{
			return _ocl_vendor;
		}

		void clPrintDevInfo(cl_device_id device);
		void clPrintAvailableDevInfo(int deviceType);
		bool clGetDeviceIdByVendorName(const char* vendor_name, cl_device_id *out);

	protected:
		CLContext();
		virtual ~CLContext();

		cl_context _context;
		cl_device_id _deviceId;
		std::string _ocl_vendor;

		static CLContext * _instance;
	};
}