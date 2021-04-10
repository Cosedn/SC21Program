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
	int dim_len[DIM] = {DIM0_LEN, DIM1_LEN, DIM2_LEN/*, DIM3_LEN, DIM4_LEN*/};
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
	int index_u[DIM0_LEN + 1], index_v[DIM1_LEN + 1];
	int in, out;
	int in_iter = NONZEROS_NUM, out_iter = 500;
//	int random[NONZEROS_NUM];
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
//	float alpha = 0.000001;
	float alpha = 0.001;
	float lamda = 0.05;
	
	std::default_random_engine generator, generator1;
	std::uniform_int_distribution<int> distribution(0, NONZEROS_NUM - 1);
	std::uniform_real_distribution<float> distribution1(0.5, 1.5);

	cudaError_t cudaStatus = cudaGetLastError();

	if((fp = fopen("../../yelp/process3/file6", "r")) == NULL)
//	if((fp = fopen("ratings10m.dat", "r")) == NULL)
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

	while((fscanf(fp, "%d %d %f %d %d %d %d", &r[size].row, &r[size].col, &r[size].rating, &temp1, &r[size].ctx, &temp2, &temp3)) != EOF) size ++;

//	printf("%d\n", size);

/*	for(i = 0; i < NONZEROS_NUM; i++) random[i] = distribution(generator);
	for(i = 0; i < 5; i++) printf("%d ", random[i]);
	printf("\n");*/

/*	for(i = 0; i < 10; i++) printf("%.2f ", U[i]);
	printf("\n");*/

	for(i = 0; i < DIM0_LEN * ATTR; i++) U[i] = distribution1(generator1);
	for(i = 0; i < DIM1_LEN * ATTR; i++) V[i] = distribution1(generator1);
	for(i = 0; i < DIM2_LEN * ATTR; i++) W[i] = distribution1(generator1);

/*	for(i = 0; i < 10; i++) printf("%.2f ", U[i]);
	printf("\n");*/

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
/*	for(i = 0; i < DIM0_LEN; i++) if(num_u[i] == 0) printf("%.2f ", num_u[i]);
	printf("\n");
	for(i = 0; i < DIM1_LEN; i++) if(num_v[i] == 0) printf("%.2f ", num_v[i]);
	printf("\n");
	for(i = 0; i < DIM2_LEN; i++) if(num_w[i] == 0) printf("%.2f ", num_w[i]);
	printf("\n");*/
//	printf("%d\n", num_u[20]);

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

/*	for(i = 0; i < 200; i++) printf("%d ", num_w[i]);
	printf("\n");*/

/*	j = 0;
	for(i = 0; i < NONZEROS_NUM; i++)
	{
		if(r[i].ctx > j)
		{
			j = r[i].ctx;
			k = i;
		}
		if(num_w[r[i].ctx] == 0) printf("%d\n", r[i].ctx);
	}
	printf("***%d  %d\n", k, j);*/

	for(i = 0; i < NONZEROS_NUM; i++) random[i] = distribution(generator);

	gettimeofday(&tv1,NULL);
//	t1 = GetTickCount();
	
	for(out = 0; out < out_iter; out++)
	{

//		for(i = 0; i < NONZEROS_NUM; i++) random[i] = distribution(generator);
		cudaMemcpy(dev_random, random, NONZEROS_NUM * sizeof(int), cudaMemcpyHostToDevice);
		__cudaErrorCheck(cudaStatus);

//		for(i = 0; i < DIM0_LEN * ATTR; i++) avg_grad_U[i] = 0; 
//		for(i = 0; i < DIM1_LEN * ATTR; i++) avg_grad_V[i] = 0;

/*		for(i = 0; i < NONZEROS_NUM; i++)
		{
			for(k = 0; k < ATTR; k++)
			{
				avg_grad_U[r[i].row * ATTR + k] += ( r0[i] * grad_V[r[i].col * ATTR + k] ) / num_u[r[i].row];
				avg_grad_V[r[i].col * ATTR + k] += ( r0[i] * grad_U[r[i].row * ATTR + k] ) / num_v[r[i].col];
			}
		}*/
		
/*		cudaMemcpy(dev_avg_grad_U, avg_grad_U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_avg_grad_V, avg_grad_V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyHostToDevice);
//		cudaMemcpy(dev_r0, r0, NONZEROS_NUM * sizeof(float), cudaMemcpyHostToDevice);
		__cudaErrorCheck(cudaStatus);*/

		//<<<1700, 192>>>, <<<6000, 192>>>
		
//		INIT_AVG_GRAD<<<9700, 1024>>>(dev_avg_grad_U, dev_avg_grad_V, dev_avg_grad_W);
		// block_num * thread_num > DIM0_LEN * ATTR
		
//		COMPUTE_AVG_GRAD_ATOMIC<<<8000, 1024>>>(dev_r, dev_r0, dev_grad_U, dev_grad_V, dev_grad_W, dev_avg_grad_U, dev_avg_grad_V, dev_avg_grad_W, dev_num_u, dev_num_v, dev_num_w, dev_random);
		// block_num * thread_num > NONZEROS_NUM
		
		SVRG_ATOMIC<<<8000, 1024>>>(dev_r, dev_r0, dev_U, dev_V, dev_W, dev_grad_U, dev_grad_V, dev_grad_W, dev_avg_grad_U, dev_avg_grad_V, dev_avg_grad_W, dev_random);
		
//		INIT_GRAD<<<9700, 1024>>>(dev_U, dev_V, dev_W, dev_grad_U, dev_grad_V, dev_grad_W);
		
		COMPUTE_R0<<<8000, 1024>>>(dev_r, dev_r0, dev_U, dev_V, dev_W);
		
		__cudaErrorCheck(cudaStatus);

		for(i = 0; i < NONZEROS_NUM; i++) random[i] = distribution(generator);

/*		for(in = 0; in < in_iter; in++)
		{
//			t = rand() % NONZEROS_NUM;
			t = in;
			e = 0;
			for(k = 0; k < ATTR; k++) e += U[r[t].row * ATTR + k] * V[r[t].col * ATTR + k];
			e = r[t].rating - e;
			
			for(k = 0; k < ATTR; k++)
			{
				temp = V[r[t].col * ATTR + k];
				grad0 = e * U[r[t].row * ATTR + k]  - lamda * V[r[t].col * ATTR + k];
				grad1 = r0[t] * grad_U[r[t].row * ATTR + k];
	 			V[r[t].col * ATTR + k] += alpha * (grad0 - grad1 + avg_grad_V[r[t].col * ATTR + k]);
				grad0 = e * temp - lamda * U[r[t].row * ATTR + k];
				grad1 = r0[t] * grad_V[r[t].col * ATTR + k];
				U[r[t].row * ATTR + k] += alpha * (grad0 - grad1 + avg_grad_U[r[t].row * ATTR + k]);
			}
		}*/
		
/*		cudaMemcpy(grad_U, dev_U, DIM0_LEN * ATTR * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(grad_V, dev_V, DIM1_LEN * ATTR * sizeof(float), cudaMemcpyDeviceToHost);*/
		cudaMemcpy(r0, dev_r0, NONZEROS_NUM * sizeof(float), cudaMemcpyDeviceToHost);
		
		__cudaErrorCheck(cudaStatus);

//		for(i = 0; i < DIM0_LEN * ATTR; i++) grad_U[i] = U[i];
//		for(i = 0; i < DIM1_LEN * ATTR; i++) grad_V[i] = V[i];

/*        for(i = 0; i < NONZEROS_NUM; i++)
        {
			r0[i] = 0;
        	for(k = 0; k < ATTR; k++) r0[i] += U[r[i].row * ATTR + k] * V[r[i].col * ATTR + k];
		}
		for(i = 0; i < NONZEROS_NUM; i++) r0[i] = r[i].rating - r0[i];*/
		
//		printf("3\n");
/*		for(i = 0; i < 100; i++) printf("%.6f ", r0[i]);
		printf("\n");*/
		curr_rmse = 0.0;
		sum = 0;
		for(i = 0; i < NONZEROS_NUM; i++) 
		{
			temp = r0[i] / NONZEROS_NUM * r0[i];
/*			if(temp > sum)
			{
				sum = temp;
				k = i;
			}*/
			curr_rmse += temp;
		}
/*		cudaMemcpy(avg_grad_W, dev_avg_grad_W, DIM2_LEN * ATTR * sizeof(float), cudaMemcpyDeviceToHost);
		for(i = 0; i < DIM2_LEN; i++) fprintf(record, "%d %.4f\n", i, avg_grad_W[i * ATTR]);
		fprintf(record, "\n\n");*/
/*			if( i > 5 && i < 10)
			{
				cudaMemcpy(&U[r[k].row * ATTR], &dev_U[r[k].row * ATTR], ATTR * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(&V[r[k].col * ATTR], &dev_V[r[k].col * ATTR], ATTR * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(&W[r[k].ctx * ATTR], &dev_W[r[k].ctx * ATTR], ATTR * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(&avg_grad_U[r[k].row * ATTR], &dev_avg_grad_U[r[k].row * ATTR], ATTR * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(&avg_grad_V[r[k].col * ATTR], &dev_avg_grad_V[r[k].col * ATTR], ATTR * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(&avg_grad_W[r[k].ctx * ATTR], &dev_avg_grad_W[r[k].ctx * ATTR], ATTR * sizeof(float), cudaMemcpyDeviceToHost);
				__cudaErrorCheck(cudaStatus);
				printf("***************\n");
				printf("%d  %.6f\n", i, sum);
				printf("%d  %d  %d\n", num_u[r[k].row], num_v[r[k].col], num_w[r[k].ctx]);
				for(j = 0; j < ATTR; j++) printf("%.6f ", U[r[k].row * ATTR + j]);
				printf("\n");
				for(j = 0; j < ATTR; j++) printf("%.6f ", V[r[k].col * ATTR + j]);
				printf("\n");
				for(j = 0; j < ATTR; j++) printf("%.6f ", W[r[k].ctx * ATTR + j]);
				printf("\n");
				for(j = 0; j < ATTR; j++) printf("%.6f ", avg_grad_U[r[k].row * ATTR + j]);
				printf("\n");
				for(j = 0; j < ATTR; j++) printf("%.6f ", avg_grad_V[r[k].col * ATTR + j]);
				printf("\n");
				for(j = 0; j < ATTR; j++) printf("%.6f ", avg_grad_W[r[k].ctx * ATTR + j]);
				printf("\n");
				printf("***************\n");
			}*/
//		}
		curr_rmse = sqrt(curr_rmse);
		fprintf(record, "%.6f\n", curr_rmse);
//		printf("%d  %.6f\n", k, sum);

		loop_counts ++;
		if(/*curr_rmse >= pre_rmse ||*/ loop_counts >= 1000) break;
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
