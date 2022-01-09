/*
理解并行 线程 的思想
*/
#include <iostream>
#include <random>
#define N 512
__global__ void add(int *a, int *b, int *c){
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *p1, int b);

int main(){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    // 分配GPU内存地址
    cudaMalloc((void **)&d_a, size); // 指向d_a地址的指针t, t的值就是指针d_a的地址；
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    // 给 a, b, c 赋初值
    a = (int *)malloc(size);
    random_ints(a, N);
    b = (int *)malloc(size);
    random_ints(b, N);
    c = (int *)malloc(size);
    // copy inputs
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // launch
    add<<<1,N>>>(d_a, d_b, d_c);
    // copy result back
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // std out
    for (int i=0;i<N;i++){
        std::cout << a[i] << "+" << b[i] << "=" << c[i] << '\n';
    }
    // clean 
    free(a); free(b); free(c);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    return 0;

}

void random_ints(int *p1, int b){
    for (int i=0 ;i < b;i = i + 1)
    {
        p1[i] = rand() % 10;
    }
}