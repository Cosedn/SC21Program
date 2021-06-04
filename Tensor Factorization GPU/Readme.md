# Tensor Factorization Algorithms

## 1 Description

This project includes the CUDA implementations of 3 sparse tensor PARAFAC(CP) decomposition algorithms: KroMagnon-TF, Hogwild!, and CUSNTF. The project include four programs and a dataset "yelp-small". Each program is in a folder. The folders are described as follows:

|  FOLDER NAME       | DESCRIPTION |
| ---------------- | ----------- |
| KroMagnon-TF_1GPU  | KroMagnon-TF paralleled with 1 GPU.  |		
| KroMagnon-TF_4GPUs | KroMagnon-TF paralleled with 4 GPUs. |
| Hogwild_1GPU       | Hogwild! paralleled with 1 GPU.  |
| CUSNTF_1GPU        | CUSNTF paralleled with 1 GPU.   |

We use the macros in Section 2.2 to describe our programs. All the four programs is to factorize a 3-order sparse tensor R of size DIM0_LEN \* DIM1_LEN \* DIM2_LEN into 3 matrices U, V and W. The sizes of U, V and W are DIM0_LEN \* ATTR, DIM1_LEN \* ATTR and DIM2_LEN \* ATTR, respectively. KroMagnon-TF is the algorithm we proposed, which accelerates gradient descent convergence using KroMagnon, a Stochastic Variance Reduced Gradient (SVRG) method. Hogwild! and CUSNTF are previous algorithms used to compare with KroMagnon-TF.

For KroMagnon-TF and Hogwild!, the programs will iterate MAX_ITER \* NONZEROS_NUM times. The stepsize of each iteration is LEARNING_RATE, and the regularization parameter LAMBDA is used to prevent overfitting. For all algorithms, each program will compute RMSE MAX_ITER times.

ALL CUDA kernel functions is executed using BLOCK_NUM blocks, each block contains THREAD_NUM threads. The computation of each CUDA thread is implemented like this:

```cuda
i = blockIdx.x * blockDim.x + threadIdx.x;

while(i < NONZEROS_NUM)
{
    Processing the i-th nonzero;
	i += gridDim.x * blockDim.x;
}
```

The sum of threads of each CUDA kernel function will be BLOCK_NUM \* THREAD_NUM. If BLOCK_NUM \* THREAD_NUM > NONZEROS_NUM, some threads will not process nonzero. BLOCK_NUM \* THREAD_NUM < NONZEROS_NUM, some threads will process more than 1 nonzeros.

The output of each program is "record" file, which includes RMSE results and "average time per loop". The "average time per loop" is the total time of iterations devided by MAX_ITER. The program currently does not output more details such as the training result of U, V, W and R. To see them, you need to write the print code yourself.

### 1.2 Platform

We test the four programs on TianHe2 GPU Platform. The hardware and software information of TianHe2 GPU Platform is as follows:

- Hardware:
  - Operating System: CentOS 7, Linux version 3.10.0-327.el7.x86_64
  - CPU: Intel Xeon E5-2660 v3 CPU 2.60 GHz, 10 cores
  - RAM：256GB
  - GPU: NVIDIA Tesla K80 GPU, 4992 CUDA cores (26 multiprocessors, 192 CUDA cores each multiprocessor), 24GB Global Memory
  - Node: Each node has 1 CPU and 2 GPUs. Each GPU has two independent devices, and each GPU device has 13 multiprocessors. Each node has 4 GPU devices.
- Software:
  - Compiler: nvcc (NVIDIA CUDA 8.0, C++ 11)

Currently we do not test our program on other platforms.

## 2 Getting Started

### 2.1 Prepare Dataset

The sparse tensor R should be a 3-order tensor. The dataset should only store nonzeros of R in COO format, i.e., each nonzero is stored as (DIM0 index | DIM1 index | DIM2 index | value).

Details of the three elements are as follows:

|NAME			    | TYPE			|    RANGE       |
| --- | :---: | --- |
|DIM0 index	  | integer	  |		0 to DIM0_LEN   |
|DIM1 index   | integer	  |		0 to DIM1_LEN   |
|DIM2 index   | integer	  |		0 to DIM2_LEN   |
|value        | double    |        >=0          |

We have provided a dataset "yelp-small" for example, and all the four programs can run directly on this dataset. If you want to try other datasets or try other parameter settings, please modify the macros in **dev.h** before compiling.

### 2.2 Set Macros

Before compiling the program, you should make sure that the macros in **dev.h** are set correctly. The macros you need to check are as follows:

|  MACRO NAME        |   MACRO VALUE TYPE  |   DESCRIPTION |
| --- | :---: | --- |
|    DIM             |        integer      | The dimension of tensor, 3 by default. This value should not be changed unless you rewrite the source code. |
|    DIM0_LEN        |        integer      | The length of DIM0. The DIM0 indexes range in [0, DIM0_LEN - 1]. |
|    DIM1_LEN        |        integer      | The length of DIM1. The DIM1 indexes range in [0, DIM1_LEN - 1]. |
|    DIM2_LEN        |        integer      | The length of DIM2. The DIM2 indexes range in [0, DIM2_LEN - 1]. |
|    NONZEROS_NUM    |        integer      | The number of nonzeros in the tensor. |
|    ATTR            |        integer      | The rank of factor matrices. |
|    LEARNING_RATE   |        float        | The learning rate of SGD. |
|    LAMBDA          |        float        | The regularization parameter. |
|    MAX_ITER        |        integer      | The maximum number of iterations. |
|    RANDOM_MIN      |        float        | The U, V, W matrices are initialized with uniformly distributed float numbers ranging in [RAND_MIN, RAND_MAX]. |
|    RANDOM_MAX      |        float        | See RAND_MIN. |
|    FILE_PATH       |        string       | The path of your dataset. |
|    BLOCK_NUM       |        integer      | The number of blocks of CUDA kernel functions. |
|    THREAD_NUM      |        integer      | The number of threads in each block. |

Here is when you need to modify these marcos:

* If you try other datasets on our programs, you need to modify DIM0_LEN, DIM1_LEN, DIM2_LEN and NONZEROS_NUM.

* If you want try different arguments to train the same dataset, you can modify ATTR, LEARNING_RATE, LAMBDA, MAX_ITER, RANDOM_MIN and RANDOM_MAX.

* If you want to improve CUDA parallel efficiecy on a different GPU, you can modify BLOCK_NUM and THREAD_NUM.

### 2.3 Compile

To compile a program, just use "make" command to execute the makefile in the folder of the program.

If the compiling is successful, an executable file "SGD.exe" will be generated.

### 2.4 Run

At present the programs can only run on a single node. To run KroMagnon-TF_1GPU, Hogwild_1GPU, and CUSNTF_1GPU, you need to make sure that at least 1 GPU device can be detected on the node. To run KroMagnon-TF_4GPUs, you need to make sure that at least 4 GPU devices can be detected on the node.

To check the GPU devices counts in a node, you can run `nvidia-smi` or `lspci | grep -i nvidia` command.

The program can be executed by specifying the task number (MPI process number) when submitted to the cluster. On TianHe2 platform, we use `yhrun` command to submit a program:

```
yhrun -N <node number> -n <task number> <executable file>
```

Since our program runs on a single node and using only one task, the submitting command becomes:

```
yhrun -N 1 -n 1 ./SGD.exe
```

If not on TianHe2 platform, the submitting command may be different.

