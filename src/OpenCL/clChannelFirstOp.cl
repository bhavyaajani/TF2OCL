// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

__kernel void call(  __global float *KIn,__global float *KOut, const unsigned int H, 
	const unsigned int W, const unsigned int D, const unsigned int C)
				{
					int index = get_global_id(0);
					
					int h = index / (W*D*C);
					int g0 = index % (W*D*C);
					int w = g0 / (D*C);
					int g1 = g0 % (D*C);
					int d = g1 / C;
					int c = g1%C;

					KOut[c*H*W*D + h*W*D + w*D + d] = KIn[index];
				}