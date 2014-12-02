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
#include "kernel_gpu.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sys/timeb.h>
#include <time.h>
#include <stdio.h>
#include <cutil_inline.h>  

#define DEBUG_TIME

int main(int argc, char** argv)
{

	int key = -1;
	CvCapture* capture = 0;
	capture = cvCaptureFromAVI(argv[1]);
	if( !capture )
	{
		fprintf(stderr,"Could not initialize...\n");
		return -1;
	}
	IplImage* input_image = NULL;
	input_image = cvQueryFrame(capture);

	if(!input_image)
	{
		printf("Bad frame \n");
		exit(0);
	}

	cvNamedWindow("IN", 1);
	cvNamedWindow("out", 1);

	// 	input_image = cvLoadImage(argv[1], CV_LOAD_IMAGE_UNCHANGED);

	int width = input_image->width;
	int height = input_image->height;

	IplImage* out_image = cvCreateImage(cvSize(width, height), input_image->depth, 1);
	int iSizeCount=width*height*sizeof(int);
	int * rnUsedModes;
	cudaMalloc((void**)&(rnUsedModes), iSizeCount);
	cudaMemset(rnUsedModes, 0, iSizeCount);
	int nM=4;
	int iElemCount =width*height * nM * sizeof(float);
	float * ucGaussian;
	float * rWeight;
	cutilSafeCall(cudaMalloc((void**)&(ucGaussian), 4*iElemCount));
	cutilSafeCall(cudaMalloc((void**)&(rWeight), iElemCount));


	if (!out_image)
	{
		std::cout << "ERROR: Failed cvCreateImage" << std::endl;
		return -1;
	}

	unsigned char* gpu_image = NULL;
	cudaError_t cuda_err = cudaMalloc((void **)(&gpu_image), (width * height * 4) * sizeof(char));
	unsigned char* output_image = NULL;
	cuda_err = cudaMalloc((void **)(&output_image), (width * height) * sizeof(char));
	if (cuda_err != cudaSuccess)
	{
		std::cout << "ERROR: Failed cudaMalloc" << std::endl;
		return -1;
	}

	printf("entrering while loop \n");	
	while(key != 1){

		IplImage* input_image_captured = cvQueryFrame(capture);
			if(!input_image_captured)
		{
			printf("Bad frame \n");
			exit(0);
		}
	    input_image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 4);
        cvCvtColor(input_image_captured, input_image, CV_BGR2BGRA);

	

		cvShowImage("IN", input_image);
	

		cuda_err = cudaMemcpy(gpu_image, input_image->imageData, (width * height * 4) * sizeof(char), cudaMemcpyHostToDevice);
		if (cuda_err != cudaSuccess)
		{
			std::cout << "ERROR: Failed cudaMemcpy" << std::endl;
			return -1;
		}

		cuda_skin((char*)gpu_image,output_image, width, height,rnUsedModes,ucGaussian,rWeight);
		cudaThreadSynchronize();

		// check for error
		cudaError_t error2 = cudaGetLastError();
		if(error2 != cudaSuccess)
		{
			// print the CUDA error message and exit
			printf("CUDA error: %s\n", cudaGetErrorString(error2));
			exit(-1);
		}

		cudaMemcpy(out_image->imageData, output_image, (width * height) * sizeof(char), cudaMemcpyDeviceToHost);
		if (cuda_err != cudaSuccess)
		{
			std::cout << "ERROR: Failed cudaMemcpy" << std::endl;
			return -1;
		}

		

 		cvShowImage("out", out_image);
		key = cvWaitKey(10);

	}

	cuda_err = cudaFree(gpu_image);
	if (cuda_err != cudaSuccess)
	{
		std::cout << "ERROR: Failed cudaFree" << std::endl;
		return -1;
	}

 	cvSaveImage("out.png", out_image);

 	cvSaveImage("In.png", input_image);

	cvReleaseImage(&input_image);
	cvReleaseImage(&out_image);

	cvDestroyWindow("IN");
	cvDestroyWindow("out");

	

	return 0;
}
