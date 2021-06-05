#include "dev.h"

/*The gradient of CUSNTF is a fraction. This function initializes the numerators (Ups) and denominators (Downs) of U, V and W.*/
__global__ void INITIALIZE_UP_DOWN(float *up_U, float *up_V, float *up_W, float *down_U, float *down_V, float *down_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = i;

	while(i < DIM0_LEN * ATTR)
	{
		up_U[i] = 0;
		down_U[i] = 0;
		i += gridDim.x * blockDim.x;
	}

	i = t;
	while(i < DIM1_LEN * ATTR)
	{
		up_V[i] = 0;
		down_V[i] = 0;
		i += gridDim.x * blockDim.x;
	}

	i = t;
	while(i < DIM2_LEN * ATTR)
	{
		up_W[i] = 0;
		down_W[i] = 0;
		i += gridDim.x * blockDim.x;
	}
}

/*Computing the Ups and Downs of U.*/
__global__ void COMPUTE_DIM0(Ratings *r, float *U, float *V, float *W, float *up_U, float *down_U)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;
	int base_U, base_V, base_W;
	float temp, temp1[ATTR], temp2[ATTR];
	float lamda = 0.05;
	
	while(i < NONZEROS_NUM)
	{
		t = i;
		base_U = r[t].row * ATTR;
		base_V = r[t].col * ATTR;
		base_W = r[t].ctx * ATTR;
		
		temp = 0;
		for(k = 0; k < ATTR; k++) temp += U[base_U + k] * V[base_V + k] * W[base_W + k];
		for(k = 0; k < ATTR; k++) temp1[k] = r[t].rating * V[base_V + k] * W[base_W + k];
		for(k = 0; k < ATTR; k++) temp2[k] = temp * (V[base_V + k] * W[base_W + k] + lamda * U[base_U + k]);
		
		for(k = 0; k < ATTR; k++) atomicAdd(&up_U[base_U + k], temp1[k]);
		for(k = 0; k < ATTR; k++) atomicAdd(&down_U[base_U + k], temp2[k]);

		i += gridDim.x * blockDim.x;
	}
}

/*Gradient Descent of DIM0*/
__global__ void CUSNTF_DIM0(float *U, float *up_U, float *down_U)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while(i < DIM0_LEN * ATTR)
	{
		if(down_U[i] != 0) U[i] = U[i] / down_U[i] * up_U[i];
		i += gridDim.x * blockDim.x;
	}
}

/*Computing the Ups and Downs of V.*/
__global__ void COMPUTE_DIM1(Ratings *r, float *U, float *V, float *W, float *up_V, float *down_V)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;
	int base_U, base_V, base_W;
	float temp, temp1[ATTR], temp2[ATTR];
	float lamda = 0.05;
	
	while(i < NONZEROS_NUM)
	{
		t = i;
		base_U = r[t].row * ATTR;
		base_V = r[t].col * ATTR;
		base_W = r[t].ctx * ATTR;
		
		temp = 0;
		for(k = 0; k < ATTR; k++) temp += U[base_U + k] * V[base_V + k] * W[base_W + k];
		for(k = 0; k < ATTR; k++) temp1[k] = r[t].rating * U[base_U + k] * W[base_W + k];
		for(k = 0; k < ATTR; k++) temp2[k] = temp * (U[base_U + k] * W[base_W + k] + lamda * V[base_V + k]);
		
		for(k = 0; k < ATTR; k++) atomicAdd(&up_V[base_V + k], temp1[k]);
		for(k = 0; k < ATTR; k++) atomicAdd(&down_V[base_V + k], temp2[k]);

		i += gridDim.x * blockDim.x;
	}
}

/*Gradient Descent of DIM1*/
__global__ void CUSNTF_DIM1(float *V, float *up_V, float *down_V)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while(i < DIM1_LEN * ATTR)
	{
		if(down_V[i] != 0) V[i] = V[i] / down_V[i] * up_V[i];
		i += gridDim.x * blockDim.x;
	}
}

/*Computing the Ups and Downs of W.*/
__global__ void COMPUTE_DIM2(Ratings *r, float *U, float *V, float *W, float *up_W, float *down_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;
	int base_U, base_V, base_W;
	float temp, temp1[ATTR], temp2[ATTR];
	float lamda = LAMBDA;
	
	while(i < NONZEROS_NUM)
	{
		t = i;
		base_U = r[t].row * ATTR;
		base_V = r[t].col * ATTR;
		base_W = r[t].ctx * ATTR;
		
		temp = 0;
		for(k = 0; k < ATTR; k++) temp += U[base_U + k] * V[base_V + k] * W[base_W + k];
		for(k = 0; k < ATTR; k++) temp1[k] = r[t].rating * U[base_U + k] * V[base_V + k];
		for(k = 0; k < ATTR; k++) temp2[k] = temp * (U[base_U + k] * V[base_V + k] + lamda * W[base_W + k]);
		
		for(k = 0; k < ATTR; k++) atomicAdd(&up_W[base_W + k], temp1[k]);
		for(k = 0; k < ATTR; k++) atomicAdd(&down_W[base_W + k], temp2[k]);

		i += gridDim.x * blockDim.x;
	}
}

/*Gradient Descent of DIM2*/
__global__ void CUSNTF_DIM2(float *W, float *up_W, float *down_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while(i < DIM2_LEN * ATTR)
	{
		if(down_W[i] != 0) W[i] = W[i] / down_W[i] * up_W[i];
		i += gridDim.x * blockDim.x;
	}
}

/*Computing training loss*/
__global__ void COMPUTE_R0(Ratings *r, float *r0, float *U, float *V, float *W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	while(i < NONZEROS_NUM)
	{
		r0[i] = 0;
		for(k = 0; k < ATTR; k++) r0[i] += U[r[i].row * ATTR + k] * V[r[i].col * ATTR + k] * W[r[i].ctx * ATTR + k];
		r0[i] = r[i].rating - r0[i];
		i += gridDim.x * blockDim.x;
	}
}
