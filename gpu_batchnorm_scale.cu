#include "blob.h"
#include "batchnorm_scale.h"
#include <stdlib.h>
#include <math.h>

__global__ void gpu_batchnorm_scale_kernel(float* in , float* out,int d,int h,int w, float* mean , float* var, float bn_eps,float* scale,float* scale_bias,bool b,bool s)
{
	//find 3-D global thread index but limit to useful threads
    unsigned int g_x = min(w-1,blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int g_y = min(h-1,blockIdx.y * blockDim.y + threadIdx.y);
    unsigned int g_z = min(d-1,blockIdx.z * blockDim.z + threadIdx.z);
    unsigned int out_index;
    
    //calculate 1-D index
    out_index = g_x + g_y*w + g_z*h*w;    

    //perform batch norm
    if(b&&s) out[out_index] = (  (in[out_index]  - mean[g_z]) / sqrtf(var[g_z] + bn_eps)   )*scale[g_z] + scale_bias[g_z] ;
    else if(s) out[out_index] = in[out_index]*scale[g_z] + scale_bias[g_z];
    else if(b) out[out_index] = (in[out_index]  - mean[g_z]) / sqrtf(var[g_z] + bn_eps);       
          
}

BLOB* gpu_batchnorm_scale(BLOB* out,float* mean,float* var, float bn_eps,float* scale, float* scale_bias,bool dobatchnorm,bool doScale)
{
    
    // limit threads in block
    int threadlim_xyz = 10;
    int size = (out->d)*sizeof(float);
    
    //calculate number of blocks needed in each direction
    dim3 Grid(out->w/threadlim_xyz+1,out->h/threadlim_xyz+1,out->d/threadlim_xyz+1);
    dim3 Block(threadlim_xyz,threadlim_xyz,threadlim_xyz);
    
    // printf("Grid : %d x %d x %d \n",out->w/threadlim_xyz+1,out->h/threadlim_xyz+1,out->d/threadlim_xyz+1 );
    // printf("Block : %d x %d x %d \n",threadlim_xyz,threadlim_xyz,threadlim_xyz );
    
    //create pointers to data to be sent to gpu
    float *device_data=NULL,*device_out=NULL,*device_mean=NULL,*device_var=NULL, *device_scale=NULL, *device_scale_bias=NULL;

    //Allocate and Write input to GPU memory.
    blob2gpu(device_data,out);
    cudaCheckError(cudaMalloc((void **) &device_out,(out->d)*(out->h)*(out->w)*sizeof(float)));

    if(dobatchnorm){
        //Allocate memory on GPU
        cudaCheckError(cudaMalloc((void **) &device_mean,size));
        cudaCheckError(cudaMalloc((void **) &device_var,size));    
    
         //Cpy data from CPU to GPU
        cudaCheckError(cudaMemcpy(device_mean, mean, size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(device_var, var, size, cudaMemcpyHostToDevice));
   
    }
    if(doScale)
    {
        //Allocate memory on GPU
        cudaCheckError(cudaMalloc((void **) &device_scale,size));
        cudaCheckError(cudaMalloc((void **) &device_scale_bias,size));
        
         //Cpy data from CPU to GPU
        cudaCheckError(cudaMemcpy(device_scale, scale, size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(device_scale_bias, scale_bias, size, cudaMemcpyHostToDevice));
    
    }

    //call kernel with 3-D grid of 3-D blocks 
    gpu_batchnorm_scale_kernel<<< Grid,Block >>>(device_data,device_out,out->d,out->h,out->w,device_mean,device_var,bn_eps,device_scale,device_scale_bias,dobatchnorm,doScale);

    cudaDeviceSynchronize();
   
     //print error if any
    cudaCheckError(cudaPeekAtLastError());

    // write the output from gpu back to cpu 
    gpu2blob(out, device_out);
    //cudaFree(device_data);

    
    if(dobatchnorm){
        //Free allocated data
        cudaFree(device_mean);
        cudaFree(device_var);
        dobatchnorm = false;
     }
    
    if(doScale){
        //Free allocated data
        cudaFree(device_scale);
        cudaFree(device_scale_bias);
        doScale = false;
     }
    

    //return blob to convolution.c
    return out;

}

