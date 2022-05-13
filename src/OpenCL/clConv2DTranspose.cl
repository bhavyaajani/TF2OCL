// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

void fetch_local_workspace(__global float *X, int oh, int ow, 
	unsigned int iH, unsigned int iW, unsigned int D, int d, unsigned int offset, local float *work)
{ 
	int index = 0;

	for (int _h = 0 ; _h < 2; _h++)
	{
		int h1 = _h + oh;
		int h2 = _h + oh + 1;
		int h3 = _h + oh + 2;
		int h4 = _h + oh + 3;

		for (int _w = 0; _w < 2; _w++)
		{
			int w = _w + ow;

			if (h1 % 2 == 1 && w % 2 == 1)
			{
				int ih = (h1 - 1) / 2;
				int iw = (w - 1) / 2;
				work[index+d] = X[ih*iW*D + iw*D + d];
			}
			else
			{
				work[index+d] = 0;
			}

			if (h2 % 2 == 1 && w % 2 == 1)
			{
				int ih = (h2 - 1) / 2;
				int iw = (w - 1) / 2;
				work[index+d+offset] = X[ih*iW*D + iw*D + d];
			}
			else
			{
				work[index+d+offset] = 0;
			}

			if (h3 % 2 == 1 && w % 2 == 1)
			{
				int ih = (h3 - 1) / 2;
				int iw = (w - 1) / 2;
				work[index+d+2*offset] = X[ih*iW*D + iw*D + d];
			}
			else
			{
				work[index+d+2*offset] = 0;
			}

			if (h4 % 2 == 1 && w % 2 == 1)
			{
				int ih = (h4 - 1) / 2;
				int iw = (w - 1) / 2;
				work[index+d+3*offset] = X[ih*iW*D + iw*D + d];
			}
			else
			{
				work[index+d+3*offset] = 0;
			}
			index += D;
		}
	}
}
__kernel void call(  __global float *X,__global float *K,__global float *B,__global float *Y,
			const unsigned int iH,const unsigned int iW,const unsigned int D, const unsigned int oH,const unsigned int oW,const unsigned int C, 
			unsigned int relu,local float* work, const unsigned int cOff, unsigned int LC, unsigned int offset)
			{
				int g0 = get_global_id(0);
				
				int gh = g0/(oW*LC);
				int g1 = g0%(oW*LC);
				int gw = g1/LC;

				unsigned int gc = get_local_id(0) + cOff;
				
				gh = gh*4;

				//Make sure we do not go out of bounds
				if (gh < oH & gw < oW & gc < C)
				{					
					int _d = get_local_id(0);
					while (_d < D)
					{
						fetch_local_workspace(X, gh, gw, iH, iW, D, _d, offset, work);
						_d += LC;//C;
					}
				}

				barrier(CLK_LOCAL_MEM_FENCE); // sync group

				if (gh < oH & gw < oW & gc < C)
				{
					const unsigned int VN = 4;
					float8 VK;
					float8 VW;
					float8 VS = (float8)(0);
					float8 VW1;
					float8 VS1 = (float8)(0);
					const unsigned int LOOP = (2*2*D) / VN;

					for(int l=0; l < LOOP; l++)
					{
						int i = VN*l;

						VK.s0 = K[C*(i + 0) + gc];VK.s1 = K[C*(i + 1) + gc];VK.s2 = K[C*(i + 2) + gc];VK.s3 = K[C*(i + 3) + gc];
						VK.s4 = VK.s0;VK.s5 = VK.s1;VK.s6 = VK.s2;VK.s7 = VK.s3;
						//VK.s4 = K[C*(i + 4) + gc];VK.s5 = K[C*(i + 5) + gc];VK.s6 = K[C*(i + 6) + gc];VK.s7 = K[C*(i + 7) + gc];
						//VK.s8 = K[C*(i + 8) + gc];VK.s9 = K[C*(i + 9) + gc];VK.sa = K[C*(i + 10) + gc];VK.sb = K[C*(i + 11) + gc];
						//VK.sc = K[C*(i + 12) + gc];VK.sd = K[C*(i + 13) + gc];VK.se = K[C*(i + 14) + gc];VK.sf = K[C*(i + 15) + gc];
						

						VW.s0 = work[i+ 0];VW.s1 = work[i+ 1];VW.s2 = work[i+ 2];VW.s3 = work[i+ 3];
						
						int j = i + offset;
						VW.s4 = work[j+ 0];VW.s5 = work[j+ 1];VW.s6 = work[j+ 2];VW.s7 = work[j+ 3];

						j = i + 2*offset;
						VW1.s0 = work[j+ 0];VW1.s1 = work[j+ 1];VW1.s2 = work[j+ 2];VW1.s3 = work[j+ 3];

						j = i + 3*offset;
						VW1.s4 = work[j+ 0];VW1.s5 = work[j+ 1];VW1.s6 = work[j+ 2];VW1.s7 = work[j+ 3];
						//VW.s4 = work[i+ 4];VW.s5 = work[i+ 5];VW.s6 = work[i+ 6];VW.s7 = work[i+ 7];
						//VW.s8 = work[i+ 8];VW.s9 = work[i+ 9];VW.sa = work[i+ 10];VW.sb = work[i+ 11];
						//VW.sc = work[i+ 12];VW.sd = work[i+ 13];VW.se = work[i+ 14];VW.sf = work[i+ 15];
						
						VS+= VK*VW;
						VS1+= VK*VW1;
						//sum += VS.s0;sum += VS.s1;sum += VS.s2;sum += VS.s3;
						//sum += VS.s4;sum += VS.s5;sum += VS.s6;sum += VS.s7;
						//sum += VS.s8;sum += VS.s9;sum += VS.sa;sum += VS.sb;
						//sum += VS.sc;sum += VS.sd;sum += VS.se;sum += VS.sf;
					}

					for(int i = LOOP*VN;i < 2*2*D; i++)
					{
						float k = K[C*i + gc];
						const float w1 = work[i];
						const float w2 = work[i+offset];
						const float w3 = work[i+2*offset];
						const float w4 = work[i+3*offset];

						VS.s0 += k*w1;
						VS.s4 += k*w2;
						VS1.s0 += k*w3;
						VS1.s4 += k*w4;
					}

					float sum1 = VS.s0+VS.s1+VS.s2+VS.s3;
					float sum2 = VS.s4+VS.s5+VS.s6+VS.s7;
					float sum3 = VS1.s0+VS1.s1+VS1.s2+VS1.s3;
					float sum4 = VS1.s4+VS1.s5+VS1.s6+VS1.s7;

					//sum += VS.s8;sum += VS.s9;sum += VS.sa;sum += VS.sb;
					//sum += VS.sc;sum += VS.sd;sum += VS.se;sum += VS.sf;

					if(B)
					{
						float b = B[gc];
						sum1 += b;
						sum2 += b;
						sum3 += b;
						sum4 += b;
					}
					
					if(relu)
					{
						if(sum1 < 0)
						  sum1 = 0;

						if(sum2 < 0)
						  sum2 = 0;

						if(sum3 < 0)
						  sum3 = 0;

						if(sum4 < 0)
						  sum4 = 0;
					}

					Y[gh*oW*C + gw*C + gc] = sum1;	

					if (gh + 1 < oH)
						Y[(gh + 1)*oW*C + gw*C + gc] = sum2;

					if (gh + 2 < oH)
						Y[(gh + 2)*oW*C + gw*C + gc] = sum3;

					if (gh + 3 < oH)
						Y[(gh + 3)*oW*C + gw*C + gc] = sum4;
				}
			}