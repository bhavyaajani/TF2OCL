// (c) 2020 Bhavya Ajani
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

#include <Tensor.h>
#include <Model.h>
#include <OpenCL\clContext.h>

void demo_run(char* json)
{
	tf2ocl::Model model;

	bool status = false;
	
	tf2ocl::CLContext::Instance()->bind("CPU");

	try
	{
		status = model.load(json);
	}
	catch (std::exception const& e)
	{
		std::cout << "Exception: " << e.what() << "\n";
	}

	int H = 224;
	int W = 224;
	int D = 3;

	tf2ocl::TensorNDim ndim; ndim.push_back(H); ndim.push_back(W); ndim.push_back(D);
	tf2ocl::Tensor input(ndim, tf2ocl::TensorDType::Float);
	input.allocateHost();

	
	float* ip = (float*)input.host();

	for (int h = 0; h < H; h++)
	{
		for (int w = 0; w < W; w++)
		{
			for (int d = 0; d < D; d++)
			{
				ip[h*W*D + w*D + d] = (h + w + d)/float(H+W+D);
			}
		}
	}

	unsigned int time = 0;

	int i = 0;
	while (i < 50) {
		i++;
		tf2ocl::Tensor* pred = NULL;

		auto m_start = std::chrono::high_resolution_clock::now();
		try
		{
			pred = model.operator()(input);
			auto m_stop = std::chrono::high_resolution_clock::now();
			time += std::chrono::duration_cast<std::chrono::milliseconds>(m_stop - m_start).count();
		}
		catch (std::exception const& e)
		{
			std::cout << "Exception: " << e.what() << "\n";
		}

		float * po = (float *)pred->host();

		double avg = 0;

		for (int i = 0; i < pred->count(); i++)
		{
			avg += po[i];
		}

		//std::cout<<"[ "<< avg/ pred->count() << " ]" << std::endl;
	}

	std::cout << "Avg Model Runtime " << " : " << time/i << " ms ---------->" << std::endl;

	std::getchar();
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout << "Expect <Modle Path> argument.";
		return -1;
	}

	demo_run(argv[1]);
}