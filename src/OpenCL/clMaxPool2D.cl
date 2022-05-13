// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

__kernel void call(  __global const float* I, __global float* O, unsigned int iH, unsigned int iW, unsigned int D, 
			unsigned int oH, unsigned int oW, unsigned int pH, unsigned int pW, unsigned int sH, unsigned int sW)
				{
					int g0 = get_global_id(0);				
					int gh = g0/(oW*D);
					int g1 = g0%(oW*D);
					int gw = g1/D;

					int d = g1%D;

					//Make sure we do not go out of bounds
					if (gh < oH & gw < oW & d < D)
					{
						unsigned int ih = gh*sH;
						unsigned int iw = gw*sW;

						float maximum = I[ih*iW*D + iw*D + d];

						for (unsigned int h = ih; h < ih + pH; h++)
						{
							for (unsigned int w = iw; w < iw + pW; w++)
							{
								float v = I[h*iW*D + w*D + d];
								if (v > maximum) {
									maximum = v;
								}
							}
						}

						O[g0] = maximum;
					}
				}