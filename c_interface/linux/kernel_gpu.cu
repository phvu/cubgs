/*
 * Copyright (c) 2009, Karl Phillip Buhr
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the name of the Author nor the names of its contributors may be
 *     used to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#define SWAP(a, b, t)	t = (a); a = (b); b = (t)
__device__ int _cudaUpdateFastBgGMM(int pixel, 
									float red, float green, float blue,
									int* pModesUsed,float fSigma, float fAlphaT,float* rWeight,float4 * ucGaussian,float fTg,float fTB,float fTb,
									float fPrune,int inpPixelCnt,int nM
									)
{
	//calculate distances to the modes (+ sort)
	//here we need to go in descending order!!!

	int pos;
	bool bFitsPDF = 0;
	int bBackground = 0;
// 	float m_fOneMinAlpha = 1 - d_GMMParams.fAlphaT;
	float m_fOneMinAlpha = 1 - fAlphaT;
	int nModes = (*pModesUsed);
 	float weight, totalWeight = 0.0f;
	
	float dR, dG, dB;
	float dist, k, sigmanew;

	//go through all modes
	for (int iModes = 0; iModes < nModes; iModes++)
	{
		pos = pixel + iModes*inpPixelCnt;
		weight = rWeight[pos];

		//fit not found yet
		if (!bFitsPDF)
		{
			//check if it belongs to some of the modes
			//calculate distance
// 			float4 cGauss = d_GMMData.ucGaussian[pos];
			float4 cGauss = ucGaussian[pos];

			dR = cGauss.x - red;
			dG = cGauss.y - green;
			dB = cGauss.z - blue;

			//check if it fits the current mode (Factor * sigma)

			//square distance -slower and less accurate
			//float maxDistance = cvSqrt(m_fTg*var);
			//if ((fabs(dR) <= maxDistance) && (fabs(dG) <= maxDistance) && (fabs(dB) <= maxDistance))
			//circle
			dist = dR*dR + dG*dG + dB*dB;

			//background? - m_fTb
			if ((totalWeight < fTB) && (dist < fTb * cGauss.w))
				bBackground = 1;

			//check fit
			if (dist < fTg * cGauss.w)
			{
				//belongs to the mode
				bFitsPDF = 1;

				//update distribution
				k = fAlphaT/weight;
				weight = m_fOneMinAlpha * weight + fPrune;
				weight += fAlphaT;
				cGauss.x -= k*(dR);
				cGauss.y -= k*(dG);
				cGauss.z -= k*(dB);

				//limit update speed for cov matrice
				//not needed
				sigmanew = cGauss.w + k*(dist - cGauss.w);

				//limit the variance
				cGauss.w = sigmanew < 4 ? 4 : 
					sigmanew > 5 * fSigma ? 5 * fSigma : sigmanew;

				ucGaussian[pos] = cGauss;

				//sort
				//all other weights are at the same place and 
				//only the matched (iModes) is higher -> just find the new place for it

				for (int iLocal = iModes; iLocal > 0; iLocal--)
				{
					int posLocal = pixel + iLocal*inpPixelCnt;
					if (weight < (rWeight[posLocal-inpPixelCnt]))
					{
						break;
					}
					else
					{
					  #define SWAP(a, b, t)	t = (a); a = (b); b = (t)
						//swap
						float tmpVal;
						float4 tmpuChar;
						SWAP(ucGaussian[posLocal],
							ucGaussian[posLocal - inpPixelCnt],
							tmpuChar);
						SWAP(rWeight[posLocal],
							rWeight[posLocal - inpPixelCnt],
							tmpVal);
					}
				}

				//belongs to the mode
			}
			else
			{
				weight = m_fOneMinAlpha * weight +fPrune;

				//check prune
				if (weight < -(fPrune))
				{
					weight = 0.0f;
					nModes--;
					//	bPrune=1;
					//break;//the components are sorted so we can skip the rest
				}
			}
			//check if it fits the current mode (2.5 sigma)
			///////
		}	//fit not found yet
		else
		{
			weight = m_fOneMinAlpha * weight + fPrune;

			if (weight < -(fPrune))
			{
				weight=0.0;
				nModes--;
				//bPrune=1;
				//break;//the components are sorted so we can skip the rest
			}
		}
		totalWeight += weight;
		rWeight[pos] = weight;
	}
	//go through all modes
	//////

	//renormalize weights
	for (int iLocal = 0; iLocal < nModes; iLocal++)
	{
		rWeight[pixel + iLocal*inpPixelCnt] /= totalWeight;
	}

	//make new mode if needed and exit
	if (!bFitsPDF)
	{
		if (nModes == nM)
		{
			//replace the weakest
		}
		else
		{
			//add a new one
			//totalWeight+=m_fAlphaT;
			//pos++;
			nModes++;
		}
		pos = pixel + (nModes-1)*inpPixelCnt;

		if (nModes == 1)
			rWeight[pos] = 1;
		else
			rWeight[pos] = fAlphaT;

		//renormalize weights
		for (int iLocal = 0; iLocal < nModes-1; iLocal++)
		{
			rWeight[pixel + iLocal*inpPixelCnt] *= m_fOneMinAlpha;
		}

		float4 cGauss;
		cGauss.x = red;
		cGauss.y = green;
		cGauss.z = blue;
		cGauss.w = fSigma;
		ucGaussian[pos] = cGauss;

		//sort
		//find the new place for it
		for (int iLocal = nModes - 1; iLocal>0; iLocal--)
		{
			int posLocal = pixel + iLocal*inpPixelCnt;
			if (fAlphaT < (rWeight[posLocal - inpPixelCnt]))
			{
				break;
			}
			else
			{
				//swap
				float4 tmpuChar;
				float tmpVal;
				SWAP(ucGaussian[posLocal],
					ucGaussian[posLocal - inpPixelCnt],
					tmpuChar);
				SWAP(rWeight[posLocal],
					rWeight[posLocal - inpPixelCnt],
					tmpVal);
			}
		}
	}

	//set the number of modes
	*pModesUsed=nModes;

	return bBackground;
}

__device__ int _cudaRemoveShadowGMM(int pixel, 
									float red, float green, float blue, 
									int nModes,float4* ucGaussian,float* rWeight,float fTau,float fTb,float fTB,int inpPixelCnt)
{
	//calculate distances to the modes (+ sort)
	//here we need to go in descending order!!!
	//	long posPixel = pixel * m_nM;
	int pos;
	float tWeight = 0;
	float numerator, denominator;

	// check all the distributions, marked as background:
	for (int iModes=0;iModes<nModes;iModes++)
	{
		pos=pixel+iModes*inpPixelCnt;
 		float4 cGauss = ucGaussian[pos];
 		float weight = rWeight[pos];
		tWeight += weight;

		numerator = red * cGauss.x + green * cGauss.y + blue * cGauss.z;
		denominator = cGauss.x * cGauss.x + cGauss.y * cGauss.y + cGauss.z * cGauss.z;
		// no division by zero allowed
		if (denominator == 0)
		{
			break;
		}
		float a = numerator / denominator;

		// if tau < a < 1 then also check the color distortion
// 		if ((a <= 1) && (a >= d_GMMParams.fTau))//m_nBeta=1
		if ((a <= 1) && (a >= fTau))//m_nBeta=1
		{
			float dR=a * cGauss.x - red;
			float dG=a * cGauss.y - green;
			float dB=a * cGauss.z - blue;

			//square distance -slower and less accurate
			//float maxDistance = cvSqrt(m_fTb*var);
			//if ((fabs(dR) <= maxDistance) && (fabs(dG) <= maxDistance) && (fabs(dB) <= maxDistance))
			//circle
			float dist=(dR*dR+dG*dG+dB*dB);
// 			if (dist<d_GMMParams.fTb*cGauss.w*a*a)
			if (dist<fTb*cGauss.w*a*a)
			{
				return 2;
			}
		}
// 		if (tWeight > d_GMMParams.fTB)
		if (tWeight > fTB)
		{
			break;
		}
	}
	return 0;
}

__device__ void _cudaReplacePixelBackgroundGMM(int pixel, uchar4* pData,float4* ucGaussian)
{
	uchar4 tmp;
 	float4 cGauss = (float4)ucGaussian[pixel];
	tmp.z = (unsigned char) cGauss.x;
	tmp.y = (unsigned char) cGauss.y;
	tmp.x = (unsigned char) cGauss.z;
	(*pData) = tmp;
}



template <int BLOCK_SIZE>
__global__ void cudaUpdateFastBgGMM(uchar4* data,unsigned char* output,int* rnUsedModes,int inpPixelCnt,int iPixelsPerThread,float4* ucGaussian,float * rWeight,float fSigma, float fAlphaT,float fTg,float fTB,float fTb,float fPrune,float fTau,int nM,bool bShadowDetection,bool bRemoveForeground)
{
__shared__ int sharedInfo[1];
	if(threadIdx.x == 0)
	{
		// the start pixel for current block
		sharedInfo[0] = (blockIdx.x * BLOCK_SIZE)*iPixelsPerThread;
	}
	__syncthreads();

	int iPxStart = sharedInfo[0] + threadIdx.x;
	int iPxEnd = min( inpPixelCnt, 
		sharedInfo[0] + (BLOCK_SIZE * iPixelsPerThread));

	uchar4* pGlobalInput = data + iPxStart;
	unsigned char* pGlobalOutput = (unsigned char *)output + iPxStart;

	int* pUsedModes = rnUsedModes + iPxStart;
	unsigned char fRed, fGreen, fBlue;
	uchar4 currentInputPx;

	for(int i = iPxStart; i < iPxEnd; i += BLOCK_SIZE)
	{
		// retrieves the color
		currentInputPx = *pGlobalInput;
		fBlue = currentInputPx.x;
		fGreen = currentInputPx.y;
		fRed = currentInputPx.z;
		pGlobalInput += BLOCK_SIZE;

		// update model + background subtract
		int result = _cudaUpdateFastBgGMM(i, fRed, fGreen, fBlue, pUsedModes,fSigma,fAlphaT,rWeight,ucGaussian,fTg,fTB,fTb,
									fPrune,inpPixelCnt,nM);
		int nMLocal = *pUsedModes;
		pUsedModes += BLOCK_SIZE;

		if (bShadowDetection)
		{
			if (!result)
			{
				result= _cudaRemoveShadowGMM(i, fRed, fGreen, fBlue, nMLocal,ucGaussian,rWeight,fTau,fTb,fTB,inpPixelCnt);
			}
		}

		switch (result)
		{
		case 0:

			//foreground
			(*pGlobalOutput) = 255;
			if (bRemoveForeground) 
			{
				_cudaReplacePixelBackgroundGMM(i, pGlobalInput-BLOCK_SIZE,ucGaussian);

			}
			break;

		case 1:

			//background
			(*pGlobalOutput) = 0;
			break;

		case 2:

			//shadow
			(*pGlobalOutput) = 128;
			if (bRemoveForeground) 
			{
				_cudaReplacePixelBackgroundGMM(i, pGlobalInput-BLOCK_SIZE,ucGaussian);
			}

			break;
		}
		pGlobalOutput += BLOCK_SIZE;
	}
}

__global__ void skin_kernel(uchar4* imagem, int width, int height)
{
// 	const int i = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
/*
	if(i < width * height)
	{
		float r=imagem[i].x;
		float g=imagem[i].y;
		float b=imagem[i].z;
		
		int pureR = 255*( r/(r+g+b));
		int pureG = 255*( g/(r+g+b));

		if( !( (pureG > lowPureG) && (pureG < highPureG) && (pureR > lowPureR) && (pureR < highPureR) ) )
		imagem[i] = make_uchar4(0, 0, 0, 0);
	}
	*/
}




extern "C" void cuda_skin(char* imagem,unsigned char* output_image, int width, int height,int* rnUsedModes,float * ucGaussian,float* rWeight)
{
  		
// Tb - the threshold - n var
  float fTb = 4*4;
// Tbf - the threshold
  float fTB = 0.9f;//1-cf from the paper 
// Tgenerate - the threshold
  float fTg = 3.0f*3.0f;//update the mode or generate new
  float fSigma= 11.0f;//sigma for the new mode
// alpha - the learning factor
  float fAlphaT=0.001f;
// complexity reduction prior constant
  float fCT=0.05f;
  int nM=4;

	//shadow
// Shadow detection
  bool bShadowDetection = 1;//turn on
  float fTau = 0.5f;// Tau - shadow threshold
  bool bRemoveForeground = 0;
  fAlphaT = 0.008f;
  float fPrune = 0.008f;
  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int nThreadsPerBlock = prop.maxThreadsPerBlock / 4;
  int nBlocksPerGrid = 256;
  int inpPixelCnt = width * height;

// number of pixels per thread must be 4k, i.e. 4, 8, 12, 16, 20...
  int iPixelsPerThread = (int)ceil( inpPixelCnt *1.0 / ( nBlocksPerGrid * nThreadsPerBlock ) );
  iPixelsPerThread =  4 * (int) ceil( iPixelsPerThread / 4.0f );
  
  nBlocksPerGrid = (int)ceil(inpPixelCnt*1.0 / ((nThreadsPerBlock) * iPixelsPerThread));

//printf("launching with %d %d \n",nBlocksPerGrid,nThreadsPerBlock);

	switch( nThreadsPerBlock )
	{
	case 8:
		cudaUpdateFastBgGMM<8><<<nBlocksPerGrid,nThreadsPerBlock>>>((uchar4 *)imagem,output_image,rnUsedModes,inpPixelCnt,iPixelsPerThread,(float4 *)ucGaussian,rWeight,fSigma, fAlphaT,fTg,fTB,fTb,fPrune,fTau,nM,bShadowDetection,bRemoveForeground);
		break;
	case 16:
		cudaUpdateFastBgGMM<16><<<nBlocksPerGrid,nThreadsPerBlock>>>((uchar4 *)imagem,output_image,rnUsedModes,inpPixelCnt,iPixelsPerThread,(float4 *)ucGaussian,rWeight,fSigma, fAlphaT,fTg,fTB,fTb,fPrune,fTau,nM,bShadowDetection,bRemoveForeground);
		break;
	case 32:
		cudaUpdateFastBgGMM<32><<<nBlocksPerGrid,nThreadsPerBlock>>>((uchar4 *)imagem,output_image,rnUsedModes,inpPixelCnt,iPixelsPerThread,(float4 *)ucGaussian,rWeight,fSigma, fAlphaT,fTg,fTB,fTb,fPrune,fTau,nM,bShadowDetection,bRemoveForeground);
		break;
	case 64:
		cudaUpdateFastBgGMM<64><<<nBlocksPerGrid,nThreadsPerBlock>>>((uchar4 *)imagem,output_image,rnUsedModes,inpPixelCnt,iPixelsPerThread,(float4 *)ucGaussian,rWeight,fSigma, fAlphaT,fTg,fTB,fTb,fPrune,fTau,nM,bShadowDetection,bRemoveForeground);
		break;
	case 128:
		cudaUpdateFastBgGMM<128><<<nBlocksPerGrid,nThreadsPerBlock>>>((uchar4 *)imagem,output_image,rnUsedModes,inpPixelCnt,iPixelsPerThread,(float4 *)ucGaussian,rWeight,fSigma, fAlphaT,fTg,fTB,fTb,fPrune,fTau,nM,bShadowDetection,bRemoveForeground);
		break;
	case 256:
		cudaUpdateFastBgGMM<256><<<nBlocksPerGrid,nThreadsPerBlock>>>((uchar4 *)imagem,output_image,rnUsedModes,inpPixelCnt,iPixelsPerThread,(float4 *)ucGaussian,rWeight,fSigma, fAlphaT,fTg,fTB,fTb,fPrune,fTau,nM,bShadowDetection,bRemoveForeground);
		break;
	case 512:
		cudaUpdateFastBgGMM<512><<<nBlocksPerGrid,nThreadsPerBlock>>>((uchar4 *)imagem,output_image,rnUsedModes,inpPixelCnt,iPixelsPerThread,(float4 *)ucGaussian,rWeight,fSigma, fAlphaT,fTg,fTB,fTb,fPrune,fTau,nM,bShadowDetection,bRemoveForeground);
		break;
	}
	
}

