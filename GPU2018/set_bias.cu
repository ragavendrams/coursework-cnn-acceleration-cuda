#include "set_bias.h"

__global__ void set_bias_kernel(float* bias,float* out,int d,int h,int w)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	int o = (x / (w*h));
	if(x < d*h*w && o < d) out[x] = bias[o];
}

BLOB* set_bias(float* bias,BLOB* out,int d,int h,int w)
{
	float *d_out, *d_bias;

	cudaCheckError(cudaMalloc((void**)&d_bias,d*sizeof(float)));
	cudaCheckError(cudaMemcpy(d_bias,bias,sizeof(float)*d,cudaMemcpyHostToDevice));

	unsigned int threadcount = d*h*w;
	int maxThreadsinBlock = 512;
	int numberofBlocks = threadcount/maxThreadsinBlock;

	if(threadcount%maxThreadsinBlock!=0) numberofBlocks+=1;

	blob2gpu(d_out,out);

	set_bias_kernel<<<numberofBlocks,maxThreadsinBlock>>>(d_bias,d_out,d,h,w);
	
	gpu2blob(out,d_out);
	//cudaMemcpy(d_out,out,sizeof(float)*d*h*w -1,cudaMemcpyDeviceToHost);
	cudaFree(d_out);
	
	return out;

	
}