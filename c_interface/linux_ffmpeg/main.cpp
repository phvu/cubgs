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
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sys/timeb.h>
#include <time.h>
#include <stdio.h>
#include <cutil_inline.h>
extern "C" { 
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <SDL.h>
#include <SDL_thread.h>
}

#define DEBUG_TIME

int main(int argc, char** argv)
{


	AVFormatContext *pFormatCtx;
	int             i, videoStream;
	AVCodecContext  *pCodecCtx;
	AVCodec         *pCodec;
	AVFrame         *pFrame;
	AVFrame         *pFrameBGRA;
	AVFrame         *pFrameGRAY;
	AVPacket        packet;
	int             frameFinished;
	int             numBytes;
	int             numBytes_out;
	uint8_t         *buffer;
	uint8_t         *buffer_out;

	SDL_Overlay     *bmp_out;
	SDL_Surface     *screen_out;
	SDL_Rect        rect_out;
	if(argc < 2) {
		printf("Please provide a movie file\n");
		return -1;
	}
	// Register all formats and codecs
	av_register_all();

	if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
		fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
		exit(1);
	}
	// Open video file
	if(av_open_input_file(&pFormatCtx, argv[1], NULL, 0, NULL)!=0)
		return -1; // Couldn't open file

	// Retrieve stream information
	if(av_find_stream_info(pFormatCtx)<0)
		return -1; // Couldn't find stream information

	// Dump information about file onto standard error
	dump_format(pFormatCtx, 0, argv[1], 0);

	// Find the first video stream
	videoStream=-1;
	for(i=0; i<pFormatCtx->nb_streams; i++)
		if(pFormatCtx->streams[i]->codec->codec_type==CODEC_TYPE_VIDEO) {
			videoStream=i;
			break;
		}
	if(videoStream==-1)
		return -1; // Didn't find a video stream

	// Get a pointer to the codec context for the video stream
	pCodecCtx=pFormatCtx->streams[videoStream]->codec;

	// Find the decoder for the video stream
	pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
	if(pCodec==NULL) {
		fprintf(stderr, "Unsupported codec!\n");
		return -1; // Codec not found
	}
	// Open codec
	if(avcodec_open(pCodecCtx, pCodec)<0)
		return -1; // Could not open codec

	// Allocate video frame
	pFrame=avcodec_alloc_frame();

	// Allocate an AVFrame structure
	pFrameBGRA=avcodec_alloc_frame();
	pFrameGRAY=avcodec_alloc_frame();
	if(pFrameBGRA==NULL)
		return -1;

	// Determine required buffer size and allocate buffer
	numBytes=avpicture_get_size(PIX_FMT_BGRA, pCodecCtx->width,
	                            pCodecCtx->height);
	buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

	numBytes_out=avpicture_get_size(PIX_FMT_GRAY8, pCodecCtx->width,
	                                pCodecCtx->height);
	buffer_out=(uint8_t *)av_malloc(numBytes_out*sizeof(uint8_t));
	//create a screen to put the video
#ifndef __DARWIN__
	screen_out = SDL_SetVideoMode(pCodecCtx->width, pCodecCtx->height, 0, 0);
#else
	screen_out = SDL_SetVideoMode(pCodecCtx->width, pCodecCtx->height, 24, 0);
#endif

	printf("checkpoint1 \n");
	// Allocate a place to put our YUV image on that screen
	bmp_out = SDL_CreateYUVOverlay(pCodecCtx->width,
	                               pCodecCtx->height,
	                               SDL_YV12_OVERLAY,
	                               screen_out);

	// Assign appropriate parts of buffer to image planes in pFrameBGRA
	// Note that pFrameBGRA is an AVFrame, but AVFrame is a superset
	// of AVPicture
	avpicture_fill((AVPicture *)pFrameBGRA, buffer, PIX_FMT_BGRA, pCodecCtx->width, pCodecCtx->height);
	avpicture_fill((AVPicture *)pFrameGRAY, buffer_out, PIX_FMT_GRAY8, pCodecCtx->width, pCodecCtx->height);

	int height=pCodecCtx->height;
	int width=pCodecCtx->width;
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

	unsigned char* gpu_image = NULL;
	cudaError_t cuda_err = cudaMalloc((void **)(&gpu_image), (width * height * 4) * sizeof(char));
	unsigned char* output_image = NULL;
	cuda_err = cudaMalloc((void **)(&output_image), (width * height) * sizeof(char));
	if (cuda_err != cudaSuccess)
	{
		std::cout << "ERROR: Failed cudaMalloc" << std::endl;
		return -1;
	}

	// create context to convert original video to BGRA
	static struct SwsContext *img_convert_ctx;
	img_convert_ctx = sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width, pCodecCtx->height, PIX_FMT_BGRA, SWS_BICUBIC, NULL, NULL, NULL);

	// create context to convert BGRA video to GRAY8
	static struct SwsContext *img_convert_ctx2_5;
	img_convert_ctx2_5 = sws_getContext(pCodecCtx->width, pCodecCtx->height,PIX_FMT_BGRA, pCodecCtx->width, pCodecCtx->height,  PIX_FMT_GRAY8, SWS_BICUBIC, NULL, NULL, NULL);

	// create context to convert GRAY8 video to YUV420P
	static struct SwsContext *img_convert_ctx3;
	img_convert_ctx3 = sws_getContext(pCodecCtx->width, pCodecCtx->height, PIX_FMT_GRAY8, pCodecCtx->width, pCodecCtx->height,  PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);


	AVPicture pict_out;
	pict_out.data[0] = bmp_out->pixels[0];
	pict_out.data[1] = bmp_out->pixels[2];
	pict_out.data[2] = bmp_out->pixels[1];

	pict_out.linesize[0] = bmp_out->pitches[0];
	pict_out.linesize[1] = bmp_out->pitches[2];
	pict_out.linesize[2] = bmp_out->pitches[1];

	rect_out.x = 2;
	rect_out.y = 2;
	rect_out.w = pCodecCtx->width;
	rect_out.h = pCodecCtx->height;

	// Read frames and save first five frames to disk
	i=0;
	bool display_output = true;
	while(av_read_frame(pFormatCtx, &packet)>=0) {
		// Is this a packet from the video stream?
		if(packet.stream_index !=videoStream) 
			continue;
		
		// Decode video frame
		avcodec_decode_video(pCodecCtx, pFrame, &frameFinished, packet.data, packet.size);

		// Did we get a video frame?
		if( ! frameFinished)
			continue;

		av_free_packet(&packet);

		//convert original video frame to BGRA (1)
		sws_scale(img_convert_ctx, (const uint8_t* const*)pFrame->data, pFrame->linesize, 0, pCodecCtx->height, pFrameBGRA->data, pFrameBGRA->linesize);

		// actual processing - background substraction in gpu
		bool extract_background = true; // for debugging purposes, set this to false
		if ( extract_background )
		{
			// copy to gpu memory one frame in BGRA format
			cuda_err = cudaMemcpy(gpu_image, buffer, (width * height * 4) * sizeof(char), cudaMemcpyHostToDevice);
#ifdef DEBUG_CUDA
			if (cuda_err != cudaSuccess)
			{
				std::cout << "ERROR: Failed cudaMemcpy" << std::endl;
				return -1;
			}
#endif			
			// extract the background
			cuda_skin((char*)gpu_image,output_image, width, height,rnUsedModes,ucGaussian,rWeight);

#ifdef DEBUG_CUDA
			cudaThreadSynchronize();

			// check for error
			cudaError_t error2 = cudaGetLastError();
			if(error2 != cudaSuccess)
			{
				// print the CUDA error message and exit
				printf("CUDA error: %s\n", cudaGetErrorString(error2));
				exit(-1);
			}
#endif			

			// copy from gpu memory to cpu in GRAY8 format
			cuda_err = cudaMemcpy(buffer_out, output_image, (width * height) * sizeof(char), cudaMemcpyDeviceToHost);
#ifdef DEBUG_CUDA
			if (cuda_err != cudaSuccess)
			{
				std::cout << "ERROR: Failed cudaMemcpy" << std::endl;
				return -1;
			}
#endif			

		}
		else
		{
			//convert BGRA video to GRAY8 (2)
			sws_scale(img_convert_ctx2_5, (const uint8_t* const*)pFrameBGRA->data, pFrameBGRA->linesize, 0, pCodecCtx->height,pFrameGRAY->data, pFrameGRAY->linesize);
		}
		
		//convert GRAY8 TO yuv420P (3)
		if (display_output)
		{
			SDL_LockYUVOverlay(bmp_out);		
			sws_scale(img_convert_ctx3, (const uint8_t* const*)pFrameGRAY->data, pFrameGRAY->linesize, 0, pCodecCtx->height, pict_out.data, pict_out.linesize);
			SDL_UnlockYUVOverlay(bmp_out);
			SDL_DisplayYUVOverlay(bmp_out, &rect_out);
		}			


	}
	cuda_err = cudaFree(gpu_image);
	if (cuda_err != cudaSuccess)
	{
		std::cout << "ERROR: Failed cudaFree" << std::endl;
		return -1;
	}

	av_free(buffer);
	av_free(buffer_out);
	av_free(pFrameBGRA);

}

