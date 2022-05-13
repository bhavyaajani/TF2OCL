// (c) 2020 Bhavya Ajani 
// Code is part of SDK for C++ inference of Tensorflow Models using OpenCL.
//All rights reserved.
//THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
//IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

void fetch_local_workspace(__global float *X, int gh, int gw, 
	unsigned int H, unsigned int W, unsigned int D, unsigned int kH, unsigned int kW, int d, local float *work)
{ 
	//Expecting kH and kW to be odd!
	int offH = kH / 2;
	int offW = kW / 2;
	int index = 0;
	int offset = kH*kW*D;

	for (int _h = -offH ; _h <= offH; _h++)
	{
		int h1 = _h + gh;
		int h2 = _h + gh + 1;
		int h3 = _h + gh + 2;
		//int h4 = _h + gh + 3;
		
		for (int _w = -offW; _w <= offW; _w++)
		{
			int w = _w + gw;

			if(h1 >= 0 & h1 < H & w >= 0 & w < W)
			{
				work[index+d] = X[h1*W*D + w*D + d];
			}
			else
			{
				work[index+d] = 0;
			}

			//if(h2 >= 0 & h2 < H & w >= 0 & w < W)
			//{
			//	work[index+d+offset] = X[h2*W*D + w*D + d];
			//}
			//else
			//{
			//	work[index+d+offset] = 0;
			//}
			//if(h3 >= 0 & h3 < H & w >= 0 & w < W)
			//{
			//	work[index+d+2*offset] = X[h3*W*D + w*D + d];
			//}
			//else
			//{
			//	work[index+d+2*offset] = 0;
			//}
			//if(h4 >= 0 & h4 < H & w >= 0 & w < W)
			//{
			//	work[index+d+3*offset] = X[h4*W*D + w*D + d];
			//}
			//else
			//{
			//	work[index+d+3*offset] = 0;
			//}
			index += D;
		}
	}
}
__kernel void call(  __global float *X,__global float *K,__global float *B,__global float *Y,
			const unsigned int iH,const unsigned int iW,const unsigned int D,const unsigned int kH,
			const unsigned int kW,const unsigned int oH,const unsigned int oW,const unsigned int C, 
			const unsigned int sH, const unsigned int sW,const unsigned int padding, unsigned int relu,local float* work,
			const unsigned int cOff, unsigned int LC)
			{
				int g0 = get_global_id(0);
				
				int gh = g0/(oW*LC);
				int g1 = g0%(oW*LC);
				int gw = g1/LC;

				unsigned int gc = get_local_id(0) + cOff;

				gh = gh * 1;
				
				//Make sure we do not go out of bounds
				//TBD:: both gh and gh + 1 < oH as dimenstions are even.
				if (gh < oH & gw < oW & gc < C)
				{
					unsigned int ih = gh*sH;
					unsigned int iw = gw*sW;

					if(padding != 1)
					{
						ih += kH/2;
						iw += kW/2;
					}
					
					int _d = get_local_id(0);
					while (_d < D)
					{
						fetch_local_workspace(X, ih, iw, iH, iW, D, kH, kW, _d, work);
						_d += LC;//C;
					}

				}

				barrier(CLK_LOCAL_MEM_FENCE); // sync group

				if (gh < oH & gw < oW & gc < C)
				{
					const unsigned int VN = 8;
					float16 VK;
					float16 VW;
					float16 VW1;
					float16 VS = (float16)(0);
					float16 VS1 = (float16)(0);
					const unsigned int offset = kH*kW*D;

					const unsigned int LOOP = (kH*kW*D) / VN;

					for(int l=0; l < LOOP; l++)
					{
						int i = VN*l;

						VK.s0 = K[C*(i + 0) + gc];VK.s1 = K[C*(i + 1) + gc];VK.s2 = K[C*(i + 2) + gc];VK.s3 = K[C*(i + 3) + gc];
						VK.s4 = K[C*(i + 4) + gc];VK.s5 = K[C*(i + 5) + gc];VK.s6 = K[C*(i + 6) + gc];VK.s7 = K[C*(i + 7) + gc];

						VK.s8 = VK.s0;VK.s9 = VK.s1;VK.sa = VK.s2;VK.sb = VK.s3;
						VK.sc = VK.s4;VK.sd = VK.s5;VK.se = VK.s6;VK.sf = VK.s7;

						//VK.s8 = K[C*(i + 8) + gc];VK.s9 = K[C*(i + 9) + gc];VK.sa = K[C*(i + 10) + gc];VK.sb = K[C*(i + 11) + gc];
						//VK.sc = K[C*(i + 12) + gc];VK.sd = K[C*(i + 13) + gc];VK.se = K[C*(i + 14) + gc];VK.sf = K[C*(i + 15) + gc];
						

						VW.s0 = work[i+ 0];VW.s1 = work[i+ 1];VW.s2 = work[i+ 2];VW.s3 = work[i+ 3];
						VW.s4 = work[i+ 4];VW.s5 = work[i+ 5];VW.s6 = work[i+ 6];VW.s7 = work[i+ 7];

						//int j = i + offset;
						//VW.s8 = work[j+ 0];VW.s9 = work[j+ 1];VW.sa = work[j+ 2];VW.sb = work[j+ 3];
						//VW.sc = work[j+ 4];VW.sd = work[j+ 5];VW.se = work[j+ 6];VW.sf = work[j+ 7];

						//j = i + 2*offset;
						//VW1.s0 = work[j+ 0];VW1.s1 = work[j+ 1];VW1.s2 = work[j+ 2];VW1.s3 = work[j+ 3];
						//VW1.s4 = work[j+ 4];VW1.s5 = work[j+ 5];VW1.s6 = work[j+ 6];VW1.s7 = work[j+ 7];

						//j = i + 3*offset;
						//VW1.s8 = work[j+ 0];VW1.s9 = work[j+ 1];VW1.sa = work[j+ 2];VW1.sb = work[j+ 3];
						//VW1.sc = work[j+ 4];VW1.sd = work[j+ 5];VW1.se = work[j+ 6];VW1.sf = work[j+ 7];

						//VW.s8 = work[i+ 8];VW.s9 = work[i+ 9];VW.sa = work[i+ 10];VW.sb = work[i+ 11];
						//VW.sc = work[i+ 12];VW.sd = work[i+ 13];VW.se = work[i+ 14];VW.sf = work[i+ 15];
						
						VS+= VK*VW;
						//VS1+= VK*VW1;

						//sum += VS.s0;sum += VS.s1;sum += VS.s2;sum += VS.s3;
						//sum += VS.s4;sum += VS.s5;sum += VS.s6;sum += VS.s7;
						//sum += VS.s8;sum += VS.s9;sum += VS.sa;sum += VS.sb;
						//sum += VS.sc;sum += VS.sd;sum += VS.se;sum += VS.sf;
					}

					for(int i = LOOP*VN;i < kH*kW*D; i++)
					{
						float k = K[C*i + gc];
						const float w1 = work[i];
						//const float w2 = work[i+offset];
						//const float w3 = work[i+2*offset];
						//const float w4 = work[i+3*offset];

						VS.s0 += k*w1;
						//VS.s8 += k*w2;
						//VS1.s0 += k*w3;
						//VS1.s8 += k*w4;
					}

					float sum1 = VS.s0+VS.s1+VS.s2+VS.s3+VS.s4+VS.s5+VS.s6+VS.s7;
					//float sum2 = VS.s8+VS.s9+VS.sa+VS.sb+VS.sc+VS.sd+VS.se+VS.sf;
					//float sum3 = VS1.s0+VS1.s1+VS1.s2+VS1.s3+VS1.s4+VS1.s5+VS1.s6+VS1.s7;
					//float sum4 = VS1.s8+VS1.s9+VS1.sa+VS1.sb+VS1.sc+VS1.sd+VS1.se+VS1.sf;
					//sum += VS.s8;sum += VS.s9;sum += VS.sa;sum += VS.sb;
					//sum += VS.sc;sum += VS.sd;sum += VS.se;sum += VS.sf;

					if(B)
					{
						float b = B[gc];
						sum1 += b;
						//sum2 += b;
						//sum3 += b;
						//sum4 += b;
					}
					
					if(relu)
					{
						if(sum1 < 0)
						  sum1 = 0;

						//if(sum2 < 0)
						//  sum2 = 0;

						//if(sum3 < 0)
						//  sum3 = 0;

						//if(sum4 < 0)
						//  sum4 = 0;
					}

					Y[gh*oW*C + gw*C + gc] = sum1;

					//if (gh + 1 < oH)
					//	Y[(gh+1)*oW*C + gw*C + gc] = sum2;

					//if (gh + 2 < oH)
					//	Y[(gh+2)*oW*C + gw*C + gc] = sum3;

					//if (gh + 3 < oH)
					//	Y[(gh+3)*oW*C + gw*C + gc] = sum4;						
				}
			}