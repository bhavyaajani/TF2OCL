// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

__kernel void call(  __global const float *ip1,__global const float *ip2, __global float *op, unsigned int C1, unsigned int C2)
				{
					unsigned int idx = get_global_id(0);
					unsigned int OUT = idx*(C1+C2);
					unsigned int IN1 = idx*C1;
					unsigned int IN2 = idx*C2;

					for(int c=0;c<C1;c++)
						op[OUT+c] = ip1[IN1+c];

					OUT = OUT + C1;
					for (int c = 0; c<C2; c++)
						op[OUT + c] = ip2[IN2 + c];
				}