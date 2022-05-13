// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

__kernel void call(  __global const float *I,__global float *O)
				{
					int index = get_global_id(0);

					float v = I[index];
					O[index] = (v>0) ? v : 0;
				}