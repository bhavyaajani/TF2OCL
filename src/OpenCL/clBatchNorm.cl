// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

__kernel void call(  __global const float *I, __global float *O, __global float *mean,__global float *variance,
					__global float *gamma, __global float *beta, unsigned int D)
				{
					int index = get_global_id(0);					
					
					int d = index%D;

					float G = 1;
					if(gamma)
						G = gamma[d];	

					float B = 0;
					if(beta)
						B = beta[d];

					O[index] = G * ( (I[index] - mean[d])/ variance[d]) + B;
				}