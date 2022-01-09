/*
这里要实现的功能是将一个元素左边三个和右边三个的值进行求和
*/
#include <iostream>
#define BLOCK_SIZE 5
#define THREAD 10
#define RADIUS 3

// GPU function
__global__ void stencil_1d(int *in, int *out){
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    /*
    threadIdx 相当于局部索引，是在 block 内的索引；
    blockIdx  相当于全局索引，指明是哪一个block;
    blockDim  就是一个 block 所能容下的 threads
    */
    int lindex = threadIdx.x + RADIUS;

    //read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = (gindex - RADIUS > 0) ? in[gindex - RADIUS] : 0; // 考虑边界溢出问题
        temp[lindex + BLOCK_SIZE] = (gindex + BLOCK_SIZE < BLOCK_SIZE * THREAD - 1) ? in[gindex + BLOCK_SIZE] : 0;
    }
    // 防止内存冲突
    __syncthreads();
    // 实施 stencil
    int result = 0;
    for (int offset = -RADIUS; offset<=RADIUS; offset++){
        result += temp[lindex+offset];
    }
    out[gindex] = result;
}

void random_ints(int *p1, int b);

int main(void){
    int *in, *out;
    int *d_in, *d_out;
    int size = BLOCK_SIZE * THREAD * sizeof(int);
    // GPU 地址分配
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    // 给 d_in 配置初值
    in = (int *)malloc(size);
    random_ints(in, BLOCK_SIZE*THREAD);
    out = (int *)malloc(size);
    // 将数值进行拷贝
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    // 进行 GPU 计算
    stencil_1d<<<BLOCK_SIZE, THREAD>>>(d_in, d_out);
    // 将数值拷贝出来
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    // 可视化
    std::cout << "Original" << '\n';
    for (int i=0;i<BLOCK_SIZE*THREAD;i++){
        std::cout << in[i] << ' ';
    }
    std::cout << '\n';
    std::cout << "Stencil_1d" << '\n';
    for (int i=0;i<BLOCK_SIZE*THREAD;i++){
        std::cout << out[i] << ' ';
    }
    //free memory
    cudaFree(d_in); cudaFree(d_out);
    free(in); free(out);

    return 0;
}

void random_ints(int *p1, int b){
    for (int i=0 ;i < b;i = i + 1)
    {
        p1[i] = rand() % 10;
    }
}