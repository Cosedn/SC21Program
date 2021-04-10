#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#define DIM 3
#define DIM0_LEN 1968703
#define DIM1_LEN 209393
#define DIM2_LEN 1254
//#define DIM3_LEN 183
//#define DIM4_LEN 169

#define NONZEROS_NUM 8021122

#define ATTR 5

typedef struct _Ratings
{
	int row;
	int col;
	int ctx;
	float rating;
}Ratings;

__global__ void INIT_AVG_GRAD(float *, float *, float *);
__global__ void COMPUTE_AVG_GRAD_ATOMIC(Ratings *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, int *);
__global__ void SVRG_ATOMIC(Ratings *, float *, float *, float *, float *, float *, float *, float *, float *, float *, float *, const int *);
__global__ void INIT_GRAD(float *, float *, float *, float *, float *, float *);
__global__ void COMPUTE_R0(Ratings *, float *, float *, float *, float *);

__global__ void INITIALIZE_UP_DOWN(float *, float *, float *, float *, float *, float *);
__global__ void COMPUTE_DIM0(Ratings *, float *, float *, float *, float *, float *);
__global__ void CUSNTF_DIM0(float *, float *, float *);
__global__ void COMPUTE_DIM1(Ratings *, float *, float *, float *, float *, float *);
__global__ void CUSNTF_DIM1(float *, float *, float *);
__global__ void COMPUTE_DIM2(Ratings *, float *, float *, float *, float *, float *);
__global__ void CUSNTF_DIM2(float *, float *, float *);
//__global__ void add(float *, float *);
