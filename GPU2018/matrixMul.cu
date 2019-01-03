#include "matrixmul.h"
#include <stdio.h>
#include <stdlib.h>

#define GRIDSIZE512 16
#define GRIDSIZE 1

__global__ void matrixMul(float * img, float * weight, float * out, int img_width, int img_height, int w_width, int w_height, int out_width, int out_height, int img_depth, int Kx, int Ky, int Sx, int Sy, int group , int out_group , int in_group) {
	
	unsigned int m = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

    #pragma unroll
    for( int g=0;g<group;g++) 
        for( int o=g*out_group;o<(g+1)*out_group;o++) 
            for( int i=g*in_group;i<(g+1)*in_group;i++) 
                for(int k = 0; k < Ky; k++) 
                   for(int l = 0; l < Kx; l++)
                        out[o*out_width*out_height + m*out_width + n] +=  
                        img[i*img_height*img_width + (m*Sy+k)*img_width + n*Sx+l] * weight[o*w_width*w_height + (i-(g*(img_depth/group)))*w_width + k*Kx + l];

}
    		
BLOB * gpu_matrixMul(BLOB* in, BLOB * w, BLOB * out, int Kx, int Ky, conv_param_t* p) {
	float *in_gpu, *w_gpu, *out_gpu;
    dim3 block, grid;
    
    // Calculates the grid size and the number of threads per block
    if(out->w * out->h > 512) {
        int threadsPerBlock = out->w/2;        
        grid = dim3(threadsPerBlock, threadsPerBlock, 1);
        block = dim3(out->w/threadsPerBlock, out->h/threadsPerBlock, 1);
    } else { // If the all BLOB fits in one block, grid size = 1
        grid = dim3(1, 1, 1);
        block = dim3(out->w, out->h, 1);
    }
    
    // Sends the BLOB for the GPU - input and weight
    blob2gpu(in_gpu, in);
    blob2gpu(w_gpu, w);

    // Allocs memory for the output of the conv in the ouput
    blob2gpu(out_gpu, out);

    int out_group = out->d/p->group;
    int in_group = in->d/p->group;
    //perform convolution

    matrixMul<<<grid, block>>> (in_gpu, w_gpu, out_gpu, in->w , in->h, w->w , w->h, out->w, out->h, in->d, Kx, Ky, p->Sx, p->Sy, p->group,out_group,in_group);
                          
    // Sends blob out back to the CPU
    gpu2blob(out, out_gpu);
    
    // Release space that holds the input on the GPU
    cudaCheckError(cudaFree(in_gpu));
    cudaCheckError(cudaFree(w_gpu));

	return out; 
}
