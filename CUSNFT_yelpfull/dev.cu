#include "dev.h"

/*__global__ void add(float *factor0, float * factor1)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < DIM1_LEN * ATTR)
	{
		factor0[i] += factor1[i];
	}
}*/

__global__ void INIT_AVG_GRAD(float *avg_grad_U, float *avg_grad_V, float *avg_grad_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < DIM0_LEN * ATTR) avg_grad_U[i] = 0;
	if(i < DIM1_LEN * ATTR) avg_grad_V[i] = 0;
	if(i < DIM2_LEN * ATTR) avg_grad_W[i] = 0;
}

__global__ void COMPUTE_AVG_GRAD_ATOMIC(Ratings *r, float *r0, float *grad_U, float *grad_V, float *grad_W, float *avg_grad_U, float *avg_grad_V, float *avg_grad_W, float *num_u, float *num_v, float *num_w, int *random)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int t, k;
	if(i < NONZEROS_NUM)
	{
			t = i;
			for(k = 0; k < ATTR; k++)
			{
				atomicAdd( &avg_grad_U[r[t].row * ATTR + k], ( r0[t] * grad_V[r[t].col * ATTR + k] * grad_W[r[t].ctx * ATTR + k] ) / num_u[r[t].row]);
				atomicAdd( &avg_grad_V[r[t].col * ATTR + k], ( r0[t] * grad_U[r[t].row * ATTR + k] * grad_W[r[t].ctx * ATTR + k] ) / num_v[r[t].col]);
				atomicAdd( &avg_grad_W[r[t].ctx * ATTR + k], ( r0[t] * grad_U[r[t].row * ATTR + k] * grad_V[r[t].col * ATTR + k] ) / num_w[r[t].ctx]);
/*				avg_grad_U[r[t].row * ATTR + k] += ( r0[t] * grad_V[r[t].col * ATTR + k] * grad_W[r[t].ctx * ATTR + k] ) / num_u[r[t].row];
				avg_grad_V[r[t].col * ATTR + k] += ( r0[t] * grad_U[r[t].row * ATTR + k] * grad_W[r[t].ctx * ATTR + k] ) / num_v[r[t].col];
				avg_grad_W[r[t].ctx * ATTR + k] += ( r0[t] * grad_U[r[t].row * ATTR + k] * grad_V[r[t].col * ATTR + k] ) / num_w[r[t].ctx];*/
/*				avg_grad_U[r[t].row * ATTR + k] = 0;
				avg_grad_V[r[t].col * ATTR + k] = 0;
				avg_grad_W[r[t].ctx * ATTR + k] = 0;*/
			}
	}
}
	
__global__ void SVRG_ATOMIC(Ratings *r, float *r0, float *U, float *V, float *W, float *grad_U, float *grad_V, float *grad_W, float *avg_grad_U, float *avg_grad_V, float *avg_grad_W, const int *random)
{
	int i, k, t;
	float e;
	float grad0, grad1;
	float temp[ATTR], temp1[ATTR], temp2[ATTR];
	float avg_temp[ATTR], avg_temp1[ATTR], avg_temp2[ATTR];
	int base_U, base_V, base_W;

	const float alpha = 0.001;
	const float lamda = 0.05;

	i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < NONZEROS_NUM)
	{
		t = random[i];
//		t = i;
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
/*			temp[k] = avg_grad_U[r[t].row * ATTR + k];
                        temp1[k] = avg_grad_V[r[t].col * ATTR + k];
                        temp2[k] = avg_grad_W[r[t].ctx * ATTR + k];*/

		for(k = 0; k < ATTR; k++) e += temp[k] * temp1[k] * temp2[k];
		e = r[t].rating - e;
			
		for(k = 0; k < ATTR; k++)
		{
			grad0 = e * temp[k] * temp1[k] - lamda * temp2[k];
			grad1 = r0[t] * grad_U[base_U + k] * grad_V[base_V + k];
//			grad1 = avg_grad_W[r[t].ctx * ATTR + k];
			grad0 = alpha * (grad0 - grad1 + avg_temp2[k] /*avg_grad_W[r[t].ctx * ATTR + k]*/);
//			grad0 = alpha * (grad0 - grad1 + avg_grad_W[r[t].ctx * ATTR + k]);
			W[base_W + k] += grad0;
//			atomicAdd( &W[r[t].ctx * ATTR + k], grad0);

			grad0 = e * temp[k] * temp2[k]  - lamda * temp1[k];
			grad1 = r0[t] * grad_U[base_U + k] * grad_W[base_W + k];
			grad0 = alpha * (grad0 - grad1 + avg_temp1[k] /*- grad1 + avg_temp1[k]*/ /*+ avg_grad_V[r[t].col * ATTR + k]*/);
//			grad0 = alpha * (grad0 - grad1 + avg_grad_V[r[t].col * ATTR + k]);
			V[base_V + k] += grad0;
//	 		atomicAdd( &V[r[t].col * ATTR + k], /*alpha **/ (grad0/* - grad1 + avg_grad_V[r[t].col * ATTR + k]*/) );
//			V[r[t].col * ATTR + k] += alpha * (grad0 - grad1 + avg_grad_V[r[t].col * ATTR + k]);
			grad0 = e * temp1[k] * temp2[k] - lamda * temp[k];
			grad1 = r0[t] * grad_V[base_V + k] * grad_W[base_W + k];
			grad0 = alpha * (grad0 - grad1 + avg_temp[k] /*+ avg_grad_U[r[t].row * ATTR + k]*/);
//			grad0 = alpha * (grad0 - grad1 + avg_grad_U[r[t].row * ATTR + k]);
			U[base_U + k] += grad0;
//			atomicAdd( &U[r[t].row * ATTR + k], /*alpha **/ (grad0/* - grad1 + avg_grad_U[r[t].row * ATTR + k]*/) );
//			U[r[t].row * ATTR + k] += alpha * (grad0 - grad1 + avg_grad_U[r[t].row * ATTR + k]);
		}
	}

}

__global__ void INIT_GRAD(float *U, float *V, float *W, float *grad_U, float *grad_V, float *grad_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < DIM0_LEN * ATTR) grad_U[i] = U[i];
	if(i < DIM1_LEN * ATTR) grad_V[i] = V[i];
	if(i < DIM2_LEN * ATTR) grad_W[i] = W[i];
}

__global__ void COMPUTE_R0(Ratings *r, float *r0, float *U, float *V, float *W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k;
	if(i < NONZEROS_NUM)
	{
		r0[i] = 0;
		for(k = 0; k < ATTR; k++) r0[i] += U[r[i].row * ATTR + k] * V[r[i].col * ATTR + k] * W[r[i].ctx * ATTR + k];
		r0[i] = r[i].rating - r0[i];
	}
}

__global__ void INITIALIZE_UP_DOWN(float *up_U, float *up_V, float *up_W, float *down_U, float *down_V, float *down_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < DIM0_LEN * ATTR)
	{
		up_U[i] = 0;
		down_U[i] = 0;
	}
	if(i < DIM1_LEN * ATTR)
	{
		up_V[i] = 0;
		down_V[i] = 0;
	}
	if(i < DIM2_LEN * ATTR)
	{
		up_W[i] = 0;
		down_W[i] = 0;
	}
}

__global__ void COMPUTE_DIM0(Ratings *r, float *U, float *V, float *W, float *up_U, float *down_U)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;
	int base_U, base_V, base_W;
	float temp, temp1[ATTR], temp2[ATTR];
	float lamda = 0.05;
	
	if(i < NONZEROS_NUM)
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
	}
}

__global__ void CUSNTF_DIM0(float *U, float *up_U, float *down_U)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, l = 1;
	if(i < DIM0_LEN)
	{
		for(k = 0; k < ATTR; k++) l *= down_U[i * ATTR + k];
		if(l != 0) for(k = 0; k < ATTR; k++) U[i * ATTR + k] = U[i * ATTR + k] / down_U[i * ATTR + k] * up_U[i * ATTR + k];
	}
}

__global__ void COMPUTE_DIM1(Ratings *r, float *U, float *V, float *W, float *up_V, float *down_V)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;
	int base_U, base_V, base_W;
	float temp, temp1[ATTR], temp2[ATTR];
	float lamda = 0.05;
	
	if(i < NONZEROS_NUM)
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
	}
}

__global__ void CUSNTF_DIM1(float *V, float *up_V, float *down_V)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, l = 1;
	if(i < DIM1_LEN)
	{
		for(k = 0; k < ATTR; k++) l *= down_V[i * ATTR + k];
		if(l != 0) for(k = 0; k < ATTR; k++) V[i * ATTR + k] = V[i * ATTR + k] / down_V[i * ATTR + k] * up_V[i * ATTR + k];
	}
}

__global__ void COMPUTE_DIM2(Ratings *r, float *U, float *V, float *W, float *up_W, float *down_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, t;
	int base_U, base_V, base_W;
	float temp, temp1[ATTR], temp2[ATTR];
	float lamda = 0.05;
	
	if(i < NONZEROS_NUM)
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
	}
}

__global__ void CUSNTF_DIM2(float *W, float *up_W, float *down_W)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k, l = 1;
	if(i < DIM2_LEN)
	{
		for(k = 0; k < ATTR; k++) l *= down_W[i * ATTR + k];
		if(l != 0) for(k = 0; k < ATTR; k++) W[i * ATTR + k] = W[i * ATTR + k] / down_W[i * ATTR + k] * up_W[i * ATTR + k];
	}
}
