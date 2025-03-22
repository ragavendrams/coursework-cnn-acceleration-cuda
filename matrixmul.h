
#ifndef MATRIXMUL_H
#define MATRIXMUL_H

#include "blob.h"
#include "convolution.h"


BLOB * gpu_matrixMul(BLOB* , BLOB * , BLOB * , int , int , conv_param_t* ) ;


#endif

