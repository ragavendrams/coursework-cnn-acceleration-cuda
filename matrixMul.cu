// #include "matrixmul.h"
// #include <stdio.h>
// #include <stdlib.h>

// __global__ void matrixMul(float * in, float * w, float * out, int in_w, int in_h, int w_w, int w_h, int w_d, int out_w, int out_h, int out_d,int in_d, int Kx, int Ky, int Sx, int Sy, int group , int out_group , int in_group, int prodZ, int prodY, int prodX) {
    
//     unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;
//     unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//     unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    
//     //Decode k and l from y
//    int k = y/Kx, l = y%Kx; 

//     //Decode m and n from z
//    int m = z/out_w, n = z%out_w;
//    //printf("k:%d l:%d x:%d y:%d z:%d \n",k,l,x,y,z);
//      float temp_in_w=2;
   
//     //Decode group,out_group and in_group from x
//    int g = x/(in_group*out_group) ;
//    int o = ((x%(in_group*out_group)) / in_group) + g*out_group; // calculate a value ranging from 0 - out_group and shift by g*out_group
//    int i = ((x%(in_group*out_group)) % in_group) + g*in_group;  // calculate a value ranging from 0 - in_group and shift by g*in_group

//     // if(/*m < out_h && n < out_w && z < prodZ && */k < Ky && l < Kx && z < prodY){//&& z < prodZ && y < prodY){
//     //     // #pragma unroll
//    	// 	    for( int g=0;g<group;g++) 
//     //            for( int o=g*out_group;o<(g+1)*out_group;o++) 
//     //                for( int i=g*in_group;i<(g+1)*in_group;i++)
//     //                    for(int m=0;m<out_h;m++)
//     //                       for(int n=0;n<out_w;n++)          
//     //                   		{// for(int k = 0; k < Ky; k++) 
//     //                      		// for(int l = 0; l < Kx; l++) 
//     //                             //printf("g:%d o:%d i:%d m:%d n:%d k:%d l:%d \n",g,o,i,m,n,k,l);
                                 
//     //                      	// }

    

//  int in_index = i * in_h * in_w + (m * Sy+k) * in_w + n*Sx+l;
//  int w_index = o * w_w * w_h + (i-g*in_group)*w_w + k*Kx + l;    
//  int out_index = o * out_w * out_h + m * out_w + n ;

// 		if(g<group && o < (g+1)*out_group && i<(g+1)*in_group && k < Ky && l < Kx && m < out_h && n < out_w ){
//       		temp_in_w = in[in_index]*w[w_index];
//           atomicAdd(&out[out_index] , temp_in_w);
//         		// out[out_index] += temp_in_w;                        
//      		}



// }
            
// BLOB * gpu_matrixMul(BLOB* in, BLOB * w, BLOB * out, int Kx, int Ky, conv_param_t* p) {

//     float *in_gpu, *w_gpu, *out_gpu;

//     dim3 block, grid;

//    // p->group = 1;

//     //Calculate input and output group size
//     int out_group = out->d/p->group;
//     int in_group = in->d/p->group;
    
//     //7D - p->group,out_group,in_group,Ky,Kx, out->h, out->w 
//       // encode p->group,out_group,in_group in x
//       // encode Ky,Kx in y
//       // encode out->h, out->w in z

//     int ProductX = p->group * out_group * in_group;
//     int ProductY = Kx*Ky;
//     int ProductZ = out->h * out->w;

//    // printf("group:%d in_group:%d out_group:%d Kx:%d Ky:%d out->w:%d out->h:%d out->d:%d\n",p->group,in_group,out_group,Kx,Ky,out->w,out->h,out->d);
 
//     // Calculates the grid size and the number of threads per block
//     int threadsperdirection = 8; //max 512 threads in a block
//     grid = dim3(1 + ProductX / threadsperdirection, 1 + ProductY/threadsperdirection  , 1 + ProductZ/ threadsperdirection);
//     block = dim3(threadsperdirection,threadsperdirection,threadsperdirection);

//     //printf("\n Kx:%d Ky:%d ProductY:%d BlockY:%d \n",Kx,Ky,ProductY,1 + ProductY/ threadsperdirection );
//     // Allocate memory and send input,output and weight to GPU
//     blob2gpu(in_gpu, in);
//     blob2gpu(w_gpu, w);
//     blob2gpu(out_gpu, out);

    

//     //perform convolution - call kernel
//     matrixMul<<<grid, block>>> (in_gpu, w_gpu, out_gpu, in->w , in->h, w->w , w->h, w->d,out->w, out->h,out->d, in->d, Kx, Ky, p->Sx, p->Sy, p->group,out_group,in_group,ProductZ,ProductY,ProductX);
                          
//     // Copy output from GPU to CPU
//     gpu2blob(out, out_gpu);


    
//       // for(int i=0;i<out->d * out->h * out->w ;i++)
//       //   printf("i:%d out[i]:%f\n",i,out->data[i] );
      
        
//     // Release space that holds the input on the GPU
//     cudaCheckError(cudaFree(in_gpu));
//     cudaCheckError(cudaFree(w_gpu));

//     return out; 
// }

#include "matrixmul.h"
#include <stdio.h>

#include <stdlib.h>

__global__ void matrixMul(float * in, float * w, float * out, int in_w, int in_h, int w_w, int w_h, int out_w, int out_h, int in_d, int Kx, int Ky, int Sx, int Sy, int group , int out_group , int in_group) {
    
    unsigned int m = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

    if(m < out_h && n < out_w){
    #pragma unroll
    for( int g=0;g<group;g++) 
        for( int o=g*out_group;o<(g+1)*out_group;o++) 
            for( int i=g*in_group;i<(g+1)*in_group;i++) 
                for(int k = 0; k < Ky; k++) 
                   for(int l = 0; l < Kx; l++)
                        out[o * out_w * out_h + m * out_w + n] += in[i * in_h * in_w + (m * Sy+k) * in_w + n*Sx+l] * w[o * w_w * w_h + (i-(g*(in_d/group)))*w_w + k*Kx + l];
    }
}
            
BLOB * gpu_matrixMul(BLOB* in, BLOB * w, BLOB * out, int Kx, int Ky, conv_param_t* p) {

    float *in_gpu, *w_gpu, *out_gpu;
    dim3 block, grid;
    
    // Calculates the grid size and the number of threads per block
    int threadsperdirection_xy = 16;
    grid = dim3(1 + out->w / threadsperdirection_xy, 1 + out->h/ threadsperdirection_xy,1);
    block = dim3(threadsperdirection_xy,threadsperdirection_xy,1);

    // Allocate memory and send input,output and weight to GPU
    blob2gpu(in_gpu, in);
    blob2gpu(w_gpu, w);
    blob2gpu(out_gpu, out);

    //Calculate input and output group size
    int out_group = out->d/p->group;
    int in_group = in->d/p->group;

    //perform convolution - call kernel
    matrixMul<<<grid, block>>> (in_gpu, w_gpu, out_gpu, in->w , in->h, w->w , w->h, out->w, out->h, in->d, Kx, Ky, p->Sx, p->Sy, p->group,out_group,in_group);
                          
    // Copy output from GPU to CPU
    gpu2blob(out, out_gpu);
    
    // for(int i=0;i<out->d * out->h * out->w ;i++)
    // {
    //   printf("i:%d out[i]:%f\n",i,out->data[i] );
    // }
    // Release space that holds the input on the GPU
    cudaCheckError(cudaFree(in_gpu));
    cudaCheckError(cudaFree(w_gpu));

    return out; 
}
