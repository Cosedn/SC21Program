#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#define DIM 3		// The tensor has 3 dimensions (modes).
#define DIM0_LEN 54999	// The length of dimension 0.
#define DIM1_LEN 17624	// The length of dimension 1.
#define DIM2_LEN 471	// The length of dimension 2.

#define NONZEROS_NUM 284622	// The number of nonzeros in the tensor.

#define ATTR 5		// The rank of factor matrices.

#define LAMBDA 0.05		// The regularization parameter.
#define MAX_ITER 1000		// Maximum number of iterations.

#define RANDOM_MIN 0.5		// Initializing factor matrices with uniformly distributed float numbers in [RANDOM_MIN, RANDOM_MAX].
#define RANDOM_MAX 1.5

#define FILE_PATH ../yelp_small	// The path of input dataset.
#define STR1(R) #R
#define STR2(R) STR1(R)

#define BLOCK_NUM 600	// Number of blocks of CUDA kernel function.
#define THREAD_NUM 1024	// Number of threads in each block of CUDA kernel function.


typedef struct _Ratings
{
	int row;
	int col;
	int ctx;
	float rating;
}Ratings;


__global__ void INITIALIZE_UP_DOWN(float *, float *, float *, float *, float *, float *);
__global__ void COMPUTE_DIM0(Ratings *, float *, float *, float *, float *, float *);
__global__ void CUSNTF_DIM0(float *, float *, float *);
__global__ void COMPUTE_DIM1(Ratings *, float *, float *, float *, float *, float *);
__global__ void CUSNTF_DIM1(float *, float *, float *);
__global__ void COMPUTE_DIM2(Ratings *, float *, float *, float *, float *, float *);
__global__ void CUSNTF_DIM2(float *, float *, float *);
__global__ void COMPUTE_R0(Ratings *, float *, float *, float *, float *);
