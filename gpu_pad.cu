#include "blob.h"
#include "gpu_pad.h"
#include <stdlib.h>


__global__ void gpu_pad(float* in , float* out, int w, int d, int h , int out_h, int out_w ,int pad)
{
	//find 1-D global thread index
    unsigned int g_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    //calculate 3D input index from 1D thread index
    int temp_z = (g_x / (w*h));
    int temp_y = (g_x % (w*h)) / w;
    int temp_x = (g_x % (w*h)) % w;

    //calculate output index using input index and pad
    unsigned int out_index = (temp_x+pad) + (temp_y+pad)*(out_w) + temp_z*(out_w)*(out_h);

    //assign output to the corresponding input 
    if(g_x < d*w*h && out_index < (d)*(out_h)*(out_w))  
    	out[out_index] = in[g_x];    
}

BLOB* pad(BLOB* in, int pad)
{
    
	//create output blob
    BLOB* out = blob_calloc(in->d, in->h+(2*pad), in->w+(2*pad));

    // limit threads in block
    int MaxThreadsinBlock = 512;

    //calculate number of blocks needed
    int TotalnumofThreadsNeeded = (in->d) * (in->h) * (in->w);
    int TotalnumofBlocksNeeded = TotalnumofThreadsNeeded / MaxThreadsinBlock;

    //increment num of block is prev calc is not exactly divisble. 
    if(TotalnumofThreadsNeeded % MaxThreadsinBlock!=0 )	TotalnumofBlocksNeeded+=1;
    
    //create pointers to input and output data to be sent to gpu
    float *device_data,*device_out;

    //Allocate and Write input and output to GPU memory.
    blob2gpu(device_data,in);
    blob2gpu(device_out,out);

    //call kernel with 1-D grid of 1-D blocks 
    gpu_pad<<< TotalnumofBlocksNeeded, MaxThreadsinBlock >>>(device_data, device_out,in->w,in->d,in->h,out->h,out->w,pad);

    //print error if any
    cudaCheckError(cudaPeekAtLastError());

    // write the output from gpu back to cpu 
    gpu2blob(out, device_out);

    //free memory  
    cudaCheckError(cudaFree(device_data));
    
    //return blob to convolution.c
    return out;

}

