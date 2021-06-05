#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//#include <time.h>
//#include <windows.h>
#include <random>
#include <thread>
#include <vector>
#include "dev.h"

inline void __cudaErrorCheck(cudaError_t cudaStatus)
{
	cudaStatus = cudaGetLastError();
    
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}

inline void thread_task(int device_id, Ratings *dev_r, float *dev_r0, float *dev_U, float *dev_V, float *dev_W, float *dev_grad_U, float *dev_grad_V, float *dev_grad_W, float *dev_avg_grad_U, float *dev_avg_grad_V, float *dev_avg_grad_W, float *dev_num_u, float *dev_num_v, float *dev_num_w, int *dev_random, float *U, float *V, float *W, float *U0, float *V0, float *W0, int *dev_r_size, float *r0, int *r_index, int *r_size, int *random)
{
	cudaError_t cudaStatus = cudaGetLastError();

	cudaSetDevice(device_id);

	/*CPU sends random number array and U, V, W to GPU.*/
	cudaMemcpy(dev_random, random, r_size[device_id] * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_U, U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_V, V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_W, W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
	__cudaErrorCheck(cudaStatus);

	/*GPU executes CUDA kernel functions.*/
	INIT_GRAD<<<BLOCK_NUM, THREAD_NUM>>>(dev_U, dev_V, dev_W, dev_grad_U, dev_grad_V, dev_grad_W);

	COMPUTE_R0<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_r0, dev_U, dev_V, dev_W, dev_r_size);

	INIT_AVG_GRAD<<<BLOCK_NUM, THREAD_NUM>>>(dev_avg_grad_U, dev_avg_grad_V, dev_avg_grad_W);

	COMPUTE_AVG_GRAD_ATOMIC<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_r0, dev_grad_U, dev_grad_V, dev_grad_W, dev_avg_grad_U, dev_avg_grad_V, dev_avg_grad_W, dev_num_u, dev_num_v, dev_num_w, dev_random, dev_r_size);

	SVRG_UPDATE<<<BLOCK_NUM, THREAD_NUM>>>(dev_r, dev_r0, dev_U, dev_V, dev_W, dev_grad_U, dev_grad_V, dev_grad_W, dev_avg_grad_U, dev_avg_grad_V, dev_avg_grad_W, dev_random, dev_r_size);

	/*CPU receives the results of RMSE and U, V, W from GPU.*/
	cudaMemcpy(&r0[r_index[device_id]], dev_r0, r_size[device_id] * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(U0, dev_U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(V0, dev_V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(W0, dev_W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyDeviceToHost);

	__cudaErrorCheck(cudaStatus);
}

inline void random_number_generate(int *r_size, int **random, std::uniform_int_distribution<int> *distribution, std::default_random_engine *generator)
{
	int i, j;
	for(i = 0; i < 4; i++) 
		for(j = 0; j < r_size[i]; j++) random[i][j] = distribution[i](generator[i]);
}

inline void matrix_add(float *U0, float *V0, float *W0, float *U1, float *V1, float *W1, float size_U, float size_V, float size_W)
{
	int i;
	for(i = 0; i < size_U; i++) U0[i] += U1[i];
	for(i = 0; i < size_V; i++) V0[i] += V1[i];
	for(i = 0; i < size_W; i++) W0[i] += W1[i];
}

inline void matrix_add_and_devide(float *U, float *V, float *W, float *U0, float *V0, float *W0, float *U1, float *V1, float *W1, float size_U, float size_V, float size_W)
{
	int i;
	for(i = 0; i < size_U; i++) U[i] = U0[i] + U1[i];
	for(i = 0; i < size_V; i++) V[i] = V0[i] + V1[i];
	for(i = 0; i < size_W; i++) W[i] = W0[i] + W1[i];
	for(i = 0; i < size_U; i++) U[i] /= 4;
	for(i = 0; i < size_V; i++) V[i] /= 4;
	for(i = 0; i < size_W; i++) W[i] /= 4;
}

int main(int argc, char *argv[])
{
	int i, j, k, l, t;
	int size = 0;
	float sum;
	FILE *fp = NULL, *record = NULL;
	int dim_len[DIM] = {DIM0_LEN, DIM1_LEN, DIM2_LEN};
	float *U, *V, *W;
	float *U0[4], *V0[4], *W0[4];
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
	int num_u[4][DIM0_LEN], num_v[4][DIM1_LEN], num_w[4][DIM2_LEN];
	float num_u_f[4][DIM0_LEN], num_v_f[4][DIM1_LEN], num_w_f[4][DIM2_LEN];
	int in, out;
	int in_iter = NONZEROS_NUM, out_iter = MAX_ITER;
	int *random[4];
	int r_index[5];
	int r_size[4];

	Ratings *dev_r[4];
	float *dev_r0[4];
	float *dev_U[4], *dev_V[4], *dev_W[4];
	float *dev_avg_grad_U[4], *dev_avg_grad_V[4], *dev_avg_grad_W[4];
	float *dev_grad_U[4], *dev_grad_V[4], *dev_grad_W[4];
	float *dev_num_u[4], *dev_num_v[4], *dev_num_w[4];
	int *dev_random[4];
	int *dev_r_size[4];

	int loop_counts = 0;
	float pre_rmse = 10000000000.0, curr_rmse = 0.0;
	
	std::default_random_engine generator[4] = {
			std::default_random_engine {10},
			std::default_random_engine {20},
			std::default_random_engine {30},
			std::default_random_engine {40}};
	std::default_random_engine generator1;
	std::uniform_real_distribution<float> distribution1(RANDOM_MIN, RANDOM_MAX);

	std::thread threads[4];
	std::thread threads1;

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

        r_index[0] = 0;
        for(i = 1; i < 4; i++)
        {
                r_index[i] = (int)((double)NONZEROS_NUM*i/4);
        }
        r_index[4] = NONZEROS_NUM;
        for(i = 0; i < 4; i++) r_size[i] = r_index[i+1] - r_index[i];

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
	for(i = 0; i < 4; i++) random[i] = (int *)calloc(r_size[i], sizeof(int));
	for(i = 0; i < 4; i++) U0[i] = (float *)calloc(DIM0_LEN * ATTR, sizeof(float));
	for(i = 0; i < 4; i++) V0[i] = (float *)calloc(DIM1_LEN * ATTR, sizeof(float));
	for(i = 0; i < 4; i++) W0[i] = (float *)calloc(DIM2_LEN * ATTR, sizeof(float));
	
	printf("1\n");

	for(i = 0; i < 4; i++)
	{
		cudaSetDevice(i);
		cudaMalloc((void **)&dev_r[i], r_size[i] * sizeof(Ratings));
		cudaMalloc((void **)&dev_r0[i], r_size[i] * sizeof(float));
		cudaMalloc((void **)&dev_U[i], DIM0_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_V[i], DIM1_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_W[i], DIM2_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_avg_grad_U[i], DIM0_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_avg_grad_V[i], DIM1_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_avg_grad_W[i], DIM2_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_grad_U[i], DIM0_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_grad_V[i], DIM1_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_grad_W[i], DIM2_LEN * ATTR * sizeof(float));
		cudaMalloc((void **)&dev_num_u[i], DIM0_LEN * sizeof(float));
		cudaMalloc((void **)&dev_num_v[i], DIM1_LEN * sizeof(float));
		cudaMalloc((void **)&dev_num_w[i], DIM2_LEN * sizeof(float));
		cudaMalloc((void **)&dev_random[i], r_size[i] * sizeof(int));
		cudaMalloc((void **)&dev_r_size[i], sizeof(int));
	}
	
	__cudaErrorCheck(cudaStatus);
	
	printf("2\n");

	while((fscanf(fp, "%d %d %d %f", &r[size].row, &r[size].col, &r[size].ctx, &r[size].rating)) != EOF) size ++;


	std::uniform_int_distribution<int> distribution[4]={
				std::uniform_int_distribution<int>(0, r_size[0] - 1),
				std::uniform_int_distribution<int>(0, r_size[1] - 1),
				std::uniform_int_distribution<int>(0, r_size[2] - 1),
				std::uniform_int_distribution<int>(0, r_size[3] - 1)};

	for(i = 0; i < 4; i++)
		for(j = 0; j < r_size[i]; j++)
			random[i][j] = distribution[i](generator[i]);

	for(i = 0; i < DIM0_LEN * ATTR; i++) U[i] = distribution1(generator1);
	for(i = 0; i < DIM1_LEN * ATTR; i++) V[i] = distribution1(generator1);
	for(i = 0; i < DIM2_LEN * ATTR; i++) W[i] = distribution1(generator1);

	for(i = 0; i < DIM0_LEN * ATTR; i++) grad_U[i] = U[i];
	for(i = 0; i < DIM1_LEN * ATTR; i++) grad_V[i] = V[i];
	for(i = 0; i < DIM2_LEN * ATTR; i++) grad_W[i] = W[i];

	for(i = 0; i < 4; i++)
	{
		for(j = 0; j < DIM0_LEN; j++) num_u[i][j] = 0;
		for(j = 0; j < DIM1_LEN; j++) num_v[i][j] = 0;
		for(j = 0; j < DIM2_LEN; j++) num_w[i][j] = 0;
	}
	for(i = 0; i < 4; i++)
		for(j = 0; j < r_size[i]; j++)
		{
			num_u[i][r[r_index[i] + j].row]++;
			num_v[i][r[r_index[i] + j].col]++;
			num_w[i][r[r_index[i] + j].ctx]++;
		}
	for(i = 0; i < 4; i++)
	{
		for(j = 0; j < DIM0_LEN; j++) num_u_f[i][j] = (float)num_u[i][j];
		for(j = 0; j < DIM1_LEN; j++) num_v_f[i][j] = (float)num_v[i][j];
		for(j = 0; j < DIM2_LEN; j++) num_w_f[i][j] = (float)num_w[i][j];
	}

	for(i = 0; i < NONZEROS_NUM; i++)
	{
		r0[i] = 0;
		for(k = 0; k < ATTR; k++) r0[i] += U[r[i].row * ATTR + k] * V[r[i].col * ATTR + k] * W[r[i].ctx * ATTR + k];
	}
	for(i = 0; i < NONZEROS_NUM; i++) r0[i] = r[i].rating - r0[i];

	curr_rmse = 0.0;
	for(i = 0; i < NONZEROS_NUM; i++) curr_rmse += r0[i] / NONZEROS_NUM * r0[i];
	curr_rmse = sqrt(curr_rmse);
//	fprintf(record, "%.6f\n", curr_rmse);

	for(i = 0; i < 4; i++)
	{
		cudaSetDevice(i);
		cudaMemcpy(dev_r[i], &r[r_index[i]], r_size[i] * sizeof(Ratings), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_r0[i], &r0[r_index[i]], r_size[i] * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_U[i], U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_V[i], V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_W[i], W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_grad_U[i], grad_U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_grad_V[i], grad_V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_grad_W[i], grad_W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_avg_grad_U[i], avg_grad_U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_avg_grad_V[i], avg_grad_V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_avg_grad_W[i], avg_grad_W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_num_u[i], num_u_f[i], DIM0_LEN * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_num_v[i], num_v_f[i], DIM1_LEN * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_num_w[i], num_w_f[i], DIM2_LEN * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_r_size[i], &r_size[i], sizeof(int), cudaMemcpyHostToDevice);
	}
	
	__cudaErrorCheck(cudaStatus);


	temp1 = DIM0_LEN * ATTR;
	temp2 = DIM1_LEN * ATTR;
	temp3 = DIM2_LEN * ATTR;

	gettimeofday(&tv1,NULL);
//	t1 = GetTickCount();
	
	for(out = 0; out < out_iter; out++)
	{

		/*Spawn the threads. For threads[0] to threads[3], each thread manages a GPU. threads1 generates random numbers.*/
		for(i = 0; i < 4; i++)
			threads[i] = std::thread(thread_task, i, dev_r[i], dev_r0[i], dev_U[i], dev_V[i], dev_W[i], dev_grad_U[i], dev_grad_V[i], dev_grad_W[i], dev_avg_grad_U[i], dev_avg_grad_V[i], dev_avg_grad_W[i], dev_num_u[i], dev_num_v[i], dev_num_w[i], dev_random[i], U, V, W, U0[i], V0[i], W0[i], dev_r_size[i], r0, r_index, r_size, random[i]);
		threads1 = std::thread(random_number_generate, r_size, random, distribution, generator);

		/*Join threads[0] to threads[3].*/
		for (auto &t: threads)
		t.join ();

		/*Averaging the copies of U, V, W in each GPU.*/
		threads[0] = std::thread(matrix_add, U0[0], V0[0], W0[0], U0[1], V0[1], W0[1], temp1, temp2, temp3);
		threads[1] = std::thread(matrix_add, U0[2], V0[2], W0[2], U0[3], V0[3], W0[3], temp1, temp2, temp3);
		threads[0].join();
		threads[1].join();
		matrix_add_and_devide(U, V, W, U0[0], V0[0], W0[0], U0[2], V0[2], W0[2], temp1, temp2, temp3);

		/*Computing training loss.*/
		curr_rmse = 0.0;
		sum = 0;
		for(i = 0; i < NONZEROS_NUM; i++) 
		{
			temp = r0[i] / NONZEROS_NUM * r0[i];
			curr_rmse += temp;
		}
		curr_rmse = sqrt(curr_rmse);
		fprintf(record, "%.6f\n", curr_rmse);

		threads1.join();

		loop_counts ++;
//		if(/*curr_rmse >= pre_rmse ||*/ loop_counts >= 5000) break;
		pre_rmse = curr_rmse;
	}
	gettimeofday(&tv2,NULL);
//	t2 = GetTickCount();

	fprintf(record, "average time per loop:%.4fms\n",((float)(tv2.tv_sec-tv1.tv_sec)*1000+(float)(tv2.tv_usec-tv1.tv_usec)/1000)/loop_counts);

	for(i = 0; i < 4; i++)
	{
		cudaFree(dev_r[i]);
		cudaFree(dev_r0[i]);
		cudaFree(dev_U[i]);
		cudaFree(dev_V[i]);
		cudaFree(dev_W[i]);
		cudaFree(dev_avg_grad_U[i]);
		cudaFree(dev_avg_grad_V[i]);
		cudaFree(dev_avg_grad_W[i]);
		cudaFree(dev_grad_U[i]);
		cudaFree(dev_grad_V[i]);
		cudaFree(dev_grad_W[i]);
		cudaFree(dev_num_u[i]);
		cudaFree(dev_num_v[i]);
		cudaFree(dev_num_w[i]);
		cudaFree(dev_random[i]);
		cudaFree(dev_r_size[i]);
	}

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
	for(i = 0; i < 4; i++) free(random[i]);
	for(i = 0; i < 4; i++) free(U0[i]);
	for(i = 0; i < 4; i++) free(V0[i]);
	for(i = 0; i < 4; i++) free(W0[i]);
	fclose(fp);
	fclose(record);
	return 0;
}
