// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

__kernel void call(  __global float *M,__global float *X, __global float *Y,
				__global float *B,const unsigned int H,const unsigned int W,local float* work)
				{
					int LW = get_local_size(1);
					int lr = get_local_id(0);
					int lc = get_local_id(1);
					int gW = get_global_size(1);
					int gr = get_global_id(0);
					int gc = get_global_id(1);
					int row =  gr;
					int span = ceil((float)W/gW);
					int col =  gc*span;
					//Make sure we do not go out of bounds
					if (row < H)
					{
						float sum = 0;
						for(int c=col;c < col+span; c++)
						{
							if(c < W)
								sum += M[row+c*H]*X[c];
								
							work[lr*LW + lc] = sum;
						}
					}
					barrier(CLK_LOCAL_MEM_FENCE); // sync group
					if(gc == 0 & row < H)
					{
						float sum = 0;
						for(int c=0;c<LW;c++){
							sum += work[lr*LW+c];
						}
						if(B){sum+=B[row];}
						Y[row] = sum;
					}
				}