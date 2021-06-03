#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//#include <time.h>
//#include <windows.h>
#include <random>
#include "dev.h"

inline void __cudaErrorCheck(cudaError_t cudaStatus)
{
	cudaStatus = cudaGetLastError();
    
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}

int main(int argc, char *argv[])
{
	int i, j, k, l, t;
	int size = 0;
	float sum;
	FILE *fp = NULL, *record = NULL;
	int dim_len[DIM] = {DIM0_LEN, DIM1_LEN, DIM2_LEN};
	float *U, *V, *W;
	float *avg_grad_U, *avg_grad_V, *avg_grad_W;
	float *grad_U, *grad_V, *grad_W;
	Ratings *r;
	float *r0; // the divergence between r and the approximation of r
	float temp;
	struct timeval tv1, tv2;
//	DWORD t1, t2;
	int temp1, temp2, temp3, temp4;
	float e;
	float grad0, grad1;
	int num_u[DIM0_LEN], num_v[DIM1_LEN], num_w[DIM2_LEN];
	float num_u_f[DIM0_LEN], num_v_f[DIM1_LEN], num_w_f[DIM2_LEN];
	int in, out;
	int in_iter = NONZEROS_NUM, out_iter = MAX_ITER;
	int *random;

	Ratings *dev_r;
	float *dev_r0;
	float *dev_U, *dev_V, *dev_W;
	float *dev_avg_grad_U, *dev_avg_grad_V, *dev_avg_grad_W;
	float *dev_grad_U, *dev_grad_V, *dev_grad_W;
	float *dev_num_u, *dev_num_v, *dev_num_w;
	int *dev_random;

	int loop_counts = 0;
	float pre_rmse = 10000000000.0, curr_rmse = 0.0;
	
	std::default_random_engine generator, generator1;
	std::uniform_int_distribution<int> distribution(0, NONZEROS_NUM - 1);
	std::uniform_real_distribution<float> distribution1(RANDOM_MIN, RANDOM_MAX);

	cudaError_t cudaStatus = cudaGetLastError();

	if((fp = fopen(STR2(FILE_PATH), "r")) == NULL)
	{
		printf("cannot open this file!\n");
		exit(0);
	}
        if((record = fopen("record", "w")) == NULL)
        {
                printf("cannot open this file!\n");
                exit(0);
        }

	U = (float *)calloc(DIM0_LEN * ATTR, sizeof(float));
	V = (float *)calloc(DIM1_LEN * ATTR, sizeof(float));
	W = (float *)calloc(DIM2_LEN * ATTR, sizeof(float));
	grad_U = (float *)calloc(DIM0_LEN * ATTR, sizeof(float));
	grad_V = (float *)calloc(DIM1_LEN * ATTR, sizeof(float));
	grad_W = (float *)calloc(DIM2_LEN * ATTR, sizeof(float));
	avg_grad_U = (float *)calloc(DIM0_LEN * ATTR, sizeof(float));
	avg_grad_V = (float *)calloc(DIM1_LEN * ATTR, sizeof(float));
	avg_grad_W = (float *)calloc(DIM2_LEN * ATTR, sizeof(float));
	r = (Ratings *)calloc(NONZEROS_NUM, sizeof(Ratings));
	r0 = (float *)calloc(NONZEROS_NUM, sizeof(float));
	random = (int *)calloc(NONZEROS_NUM, sizeof(int));
	
	printf("1\n");

	cudaMalloc((void **)&dev_r, NONZEROS_NUM * sizeof(Ratings));
	cudaMalloc((void **)&dev_r0, NONZEROS_NUM * sizeof(float));
	cudaMalloc((void **)&dev_U, DIM0_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_V, DIM1_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_W, DIM2_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_avg_grad_U, DIM0_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_avg_grad_V, DIM1_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_avg_grad_W, DIM2_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_grad_U, DIM0_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_grad_V, DIM1_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_grad_W, DIM2_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_num_u, DIM0_LEN * sizeof(float));
	cudaMalloc((void **)&dev_num_v, DIM1_LEN * sizeof(float));
	cudaMalloc((void **)&dev_num_w, DIM2_LEN * sizeof(float));
	cudaMalloc((void **)&dev_random, NONZEROS_NUM * sizeof(int));
	
	__cudaErrorCheck(cudaStatus);
	
	printf("2\n");

	while((fscanf(fp, "%d %d %d %f", &r[size].row, &r[size].col, &r[size].ctx, &r[size].rating)) != EOF) size ++;


	for(i = 0; i < DIM0_LEN * ATTR; i++) U[i] = distribution1(generator1);
	for(i = 0; i < DIM1_LEN * ATTR; i++) V[i] = distribution1(generator1);
	for(i = 0; i < DIM2_LEN * ATTR; i++) W[i] = distribution1(generator1);


	for(i = 0; i < DIM0_LEN * ATTR; i++) grad_U[i] = U[i];
	for(i = 0; i < DIM1_LEN * ATTR; i++) grad_V[i] = V[i];
	for(i = 0; i < DIM2_LEN * ATTR; i++) grad_W[i] = W[i];

	for(i = 0; i < DIM0_LEN; i++) num_u[i] = 0;
	for(i = 0; i < DIM1_LEN; i++) num_v[i] = 0;
	for(i = 0; i < DIM2_LEN; i++) num_w[i] = 0;
	for(i = 0; i < NONZEROS_NUM; i++)
	{
		num_u[r[i].row]++;
		num_v[r[i].col]++;
		num_w[r[i].ctx]++;
	}

	for(i = 0; i < DIM0_LEN; i++) num_u_f[i] = (float)num_u[i];
	for(i = 0; i < DIM1_LEN; i++) num_v_f[i] = (float)num_v[i];
	for(i = 0; i < DIM2_LEN; i++) num_w_f[i] = (float)num_w[i];

	for(i = 0; i < NONZEROS_NUM; i++)
	{
		r0[i] = 0;
		for(k = 0; k < ATTR; k++) r0[i] += U[r[i].row * ATTR + k] * V[r[i].col * ATTR + k] * W[r[i].ctx * ATTR + k];
	}
	for(i = 0; i < NONZEROS_NUM; i++) r0[i] = r[i].rating - r0[i];

	curr_rmse = 0.0;
	for(i = 0; i < NONZEROS_NUM; i++) curr_rmse += r0[i] / NONZEROS_NUM * r0[i];
	curr_rmse = sqrt(curr_rmse);
	fprintf(record, "%.6f\n", curr_rmse);
	
	cudaMemcpy(dev_r, r, NONZEROS_NUM * sizeof(Ratings), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r0, r0, NONZEROS_NUM * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_U, U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_V, V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_W, W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_grad_U, grad_U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_grad_V, grad_V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_grad_W, grad_W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_num_u, num_u_f, DIM0_LEN * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_num_v, num_v_f, DIM1_LEN * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_num_w, num_w_f, DIM2_LEN * sizeof(float), cudaMemcpyHostToDevice);
	
	__cudaErrorCheck(cudaStatus);


	for(i = 0; i < NONZEROS_NUM; i++) random[i] = distribution(generator);

	gettimeofday(&tv1,NULL);
//	t1 = GetTickCount();
	
	for(out = 0; out < out_iter; out++)
	{

		/*CPU sends random number array to GPU.*/
		cudaMemcpy(dev_random, random, NONZEROS_NUM * sizeof(int), cudaMemcpyHostToDevice);
		__cudaErrorCheck(cudaStatus);

		/*GPU executes CUDA kernel functions.*/
		
		SGD_UPDATE<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_r0, dev_U, dev_V, dev_W, dev_grad_U, dev_grad_V, dev_grad_W, dev_avg_grad_U, dev_avg_grad_V, dev_avg_grad_W, dev_random);
		
		COMPUTE_R0<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_r0, dev_U, dev_V, dev_W);
		
		__cudaErrorCheck(cudaStatus);

		/*CPU generates random number array for the next outer loop while GPU runs CUDA kernel functions of the current outer loop.*/
		for(i = 0; i < NONZEROS_NUM; i++) random[i] = distribution(generator);

		/*CPU receives RMSE results from GPU.*/
		cudaMemcpy(r0, dev_r0, NONZEROS_NUM * sizeof(float), cudaMemcpyDeviceToHost);
		
		__cudaErrorCheck(cudaStatus);

		
		curr_rmse = 0.0;
		sum = 0;
		for(i = 0; i < NONZEROS_NUM; i++) 
		{
			temp = r0[i] / NONZEROS_NUM * r0[i];
			curr_rmse += temp;
		}
		curr_rmse = sqrt(curr_rmse);
		fprintf(record, "%.6f\n", curr_rmse);

		loop_counts ++;
//		if(/*curr_rmse >= pre_rmse ||*/ loop_counts >= 5000) break;
		pre_rmse = curr_rmse;
	}
	gettimeofday(&tv2,NULL);
//	t2 = GetTickCount();

	fprintf(record, "average time per loop:%.4fms\n",((float)(tv2.tv_sec-tv1.tv_sec)*1000+(float)(tv2.tv_usec-tv1.tv_usec)/1000)/loop_counts);
//	fprintf(record, "average time per loop:%.4fms\n",(float)(t2 - t1)/loop_counts);

	cudaFree(dev_r);
	cudaFree(dev_r0);
	cudaFree(dev_U);
	cudaFree(dev_V);
	cudaFree(dev_W);
	cudaFree(dev_avg_grad_U);
	cudaFree(dev_avg_grad_V);
	cudaFree(dev_avg_grad_W);
	cudaFree(dev_grad_U);
	cudaFree(dev_grad_V);
	cudaFree(dev_grad_W);
	cudaFree(dev_num_u);
	cudaFree(dev_num_v);
	cudaFree(dev_num_w);
	cudaFree(dev_random);

	free(U);
	free(V);
	free(W);
	free(grad_U);
	free(grad_V);
	free(grad_W);
	free(avg_grad_U);
	free(avg_grad_V);
	free(avg_grad_W);
	free(r);
	free(r0);
	free(random);
	fclose(fp);
	fclose(record);
	return 0;
}
