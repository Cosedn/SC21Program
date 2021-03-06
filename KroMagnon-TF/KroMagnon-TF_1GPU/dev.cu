#include "dev.h"


/*Initializing avg_grad_U, avg_grad_V and avg_grad_W with 0*/
__global__ void INIT_AVG_GRAD(float *avg_grad_U, float *avg_grad_V, float *avg_grad_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = i;

	while(i < DIM0_LEN * ATTR)
	{
		avg_grad_U[i] = 0;
		i += gridDim.x * blockDim.x;
	}

	i = t;
	while(i < DIM1_LEN * ATTR)
	{
		avg_grad_V[i] = 0;
		i += gridDim.x * blockDim.x;
	}

	i = t;
	while(i < DIM2_LEN * ATTR)
	{
		avg_grad_W[i] = 0;
		i += gridDim.x * blockDim.x;
	}
}

/*Computing avg_grad_U, avg_grad_V and avg_grad_W atomically.*/
__global__ void COMPUTE_AVG_GRAD_ATOMIC(Ratings *r, float *r0, float *grad_U, float *grad_V, float *grad_W, float *avg_grad_U, float *avg_grad_V, float *avg_grad_W, float *num_u, float *num_v, float *num_w, int *random)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t, k;

	while(i < NONZEROS_NUM)
	{
		t = i;
		for(k = 0; k < ATTR; k++)
		{
			atomicAdd( &avg_grad_U[r[t].row * ATTR + k], ( r0[t] * grad_V[r[t].col * ATTR + k] * grad_W[r[t].ctx * ATTR + k] ) / num_u[r[t].row]);
			atomicAdd( &avg_grad_V[r[t].col * ATTR + k], ( r0[t] * grad_U[r[t].row * ATTR + k] * grad_W[r[t].ctx * ATTR + k] ) / num_v[r[t].col]);
			atomicAdd( &avg_grad_W[r[t].ctx * ATTR + k], ( r0[t] * grad_U[r[t].row * ATTR + k] * grad_V[r[t].col * ATTR + k] ) / num_w[r[t].ctx]);
		}

		i += gridDim.x * blockDim.x;
	}
}

/*Updating factored matrices with KroMagnon-TF lock-freely.*/	
__global__ void SVRG_UPDATE(Ratings *r, float *r0, float *U, float *V, float *W, float *grad_U, float *grad_V, float *grad_W, float *avg_grad_U, float *avg_grad_V, float *avg_grad_W, const int *random)
{
	int i, k, t;
	float e;
	float grad0, grad1;
	float grad01, grad02, grad03;
	float temp[ATTR], temp1[ATTR], temp2[ATTR];
	float avg_temp[ATTR], avg_temp1[ATTR], avg_temp2[ATTR];
	int base_U, base_V, base_W;

	const float alpha = LEARNING_RATE;
	const float lamda = LAMBDA;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	while(i < NONZEROS_NUM)
	{
		t = random[i];
		e = 0;

		base_U = r[t].row * ATTR;
		base_V = r[t].col * ATTR;
		base_W = r[t].ctx * ATTR;
		for(k = 0; k < ATTR; k++) temp[k] = U[base_U + k];
		for(k = 0; k < ATTR; k++) temp1[k] = V[base_V + k];
		for(k = 0; k < ATTR; k++) temp2[k] = W[base_W + k];
		for(k = 0; k < ATTR; k++) avg_temp[k] = avg_grad_U[base_U + k];
		for(k = 0; k < ATTR; k++) avg_temp1[k] = avg_grad_V[base_V + k];
		for(k = 0; k < ATTR; k++) avg_temp2[k] = avg_grad_W[base_W + k];

		for(k = 0; k < ATTR; k++) e += temp[k] * temp1[k] * temp2[k];
		e = r[t].rating - e;
			
		for(k = 0; k < ATTR; k++)
		{
			grad0 = e * temp[k] * temp1[k] - lamda * temp2[k];
			grad1 = r0[t] * grad_U[base_U + k] * grad_V[base_V + k];
			grad01 = alpha * (grad0 - grad1 + avg_temp2[k]);

			grad0 = e * temp[k] * temp2[k]  - lamda * temp1[k];
			grad1 = r0[t] * grad_U[base_U + k] * grad_W[base_W + k];
			grad02 = alpha * (grad0 - grad1 + avg_temp1[k]);
			
			grad0 = e * temp1[k] * temp2[k] - lamda * temp[k];
			grad1 = r0[t] * grad_V[base_V + k] * grad_W[base_W + k];
			grad03 = alpha * (grad0 - grad1 + avg_temp[k]);

			W[base_W + k] += grad01;
			V[base_V + k] += grad02;
			U[base_U + k] += grad03;
		}

		i += gridDim.x * blockDim.x;
	}

}

/*Storing the result of U, V, W to grad_U, grad_V, grad_W as a sample*/
__global__ void INIT_GRAD(float *U, float *V, float *W, float *grad_U, float *grad_V, float *grad_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t = i;

	while(i < DIM0_LEN * ATTR)
	{
		grad_U[i] = U[i];
		i += gridDim.x * blockDim.x;
	}

	i = t;
	while(i < DIM1_LEN * ATTR)
	{
		grad_V[i] = V[i];
		i += gridDim.x * blockDim.x;
	}

	i = t;
	while(i < DIM2_LEN * ATTR)
	{
		grad_W[i] = W[i];
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
