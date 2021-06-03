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
	Ratings *r;
	float *r0; // the divergence between r and the approximation of r
	float temp;
	struct timeval tv1, tv2;
//	DWORD t1, t2;
	int temp1, temp2, temp3, temp4;
	float e;
	float grad0, grad1;
	int num_u[DIM0_LEN], num_v[DIM1_LEN], num_w[DIM2_LEN];
	int in, out;
	int in_iter = NONZEROS_NUM, out_iter = MAX_ITER;

	Ratings *dev_r;
	float *dev_r0;
	float *dev_U, *dev_V, *dev_W;
	float *dev_up_U, *dev_up_V, *dev_up_W;
	float *dev_down_U, *dev_down_V, *dev_down_W;

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
	r = (Ratings *)calloc(NONZEROS_NUM, sizeof(Ratings));
	r0 = (float *)calloc(NONZEROS_NUM, sizeof(float));
	
	printf("1\n");

	cudaMalloc((void **)&dev_r, NONZEROS_NUM * sizeof(Ratings));
	cudaMalloc((void **)&dev_r0, NONZEROS_NUM * sizeof(float));
	cudaMalloc((void **)&dev_U, DIM0_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_V, DIM1_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_W, DIM2_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_up_U, DIM0_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_up_V, DIM1_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_up_W, DIM2_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_down_U, DIM0_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_down_V, DIM1_LEN * ATTR * sizeof(float));
	cudaMalloc((void **)&dev_down_W, DIM2_LEN * ATTR * sizeof(float));
	
	__cudaErrorCheck(cudaStatus);
	
	printf("2\n");

	while((fscanf(fp, "%d %d %d %f", &r[size].row, &r[size].col, &r[size].ctx, &r[size].rating)) != EOF) size ++;


	for(i = 0; i < DIM0_LEN * ATTR; i++) U[i] = distribution1(generator1);
	for(i = 0; i < DIM1_LEN * ATTR; i++) V[i] = distribution1(generator1);
	for(i = 0; i < DIM2_LEN * ATTR; i++) W[i] = distribution1(generator1);

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
	
	__cudaErrorCheck(cudaStatus);

	printf("3\n");

	gettimeofday(&tv1,NULL);
//	t1 = GetTickCount();
	
	for(out = 0; out < out_iter; out++)
	{

		INITIALIZE_UP_DOWN<<<BLOCK_NUM, THREAD_NUM>>>(dev_up_U, dev_up_V, dev_up_W, dev_down_U, dev_down_V, dev_down_W);

		COMPUTE_DIM0<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_U, dev_V, dev_W, dev_up_U, dev_down_U);

		CUSNTF_DIM0<<<BLOCK_NUM, THREAD_NUM>>>(dev_U, dev_up_U, dev_down_U);
		
		COMPUTE_DIM1<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_U, dev_V, dev_W, dev_up_V, dev_down_V);
		
		CUSNTF_DIM1<<<BLOCK_NUM, THREAD_NUM>>>(dev_V, dev_up_V, dev_down_V);
		
		COMPUTE_DIM2<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_U, dev_V, dev_W, dev_up_W, dev_down_W);
		
		CUSNTF_DIM2<<<BLOCK_NUM, THREAD_NUM>>>(dev_W, dev_up_W, dev_down_W);
		
		COMPUTE_R0<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_r0, dev_U, dev_V, dev_W);
		
		__cudaErrorCheck(cudaStatus);

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
//		if(/*curr_rmse >= pre_rmse ||*/ loop_counts >= 1000) break;
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
	cudaFree(dev_up_U);
	cudaFree(dev_up_V);
	cudaFree(dev_up_W);
	cudaFree(dev_down_U);
	cudaFree(dev_down_V);
	cudaFree(dev_down_W);

	free(U);
	free(V);
	free(W);
	free(r);
	free(r0);
	fclose(fp);
	fclose(record);
	return 0;
}
