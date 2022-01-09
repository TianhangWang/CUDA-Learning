/*
理解并行 block 的思想
*/
#include <iostream>
#include <random>
#define N 512
__global__ void add(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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
    /*
    对于该二级指针的理解：
        首先GPU的参数传递是依靠 pass by reference 的，所以当在 CPU 中声明了一个空指针 d_a 用于 GPU 内存管理，
        那么只有转递指向d_a的二级指针d_a_p，才能获取d_a的地址管理权限
        如果只使用 (void *)d_a, 只能修改到空指针 d_a 所对应的值
                    地址         值
        d_a       0x001f      unknown
        d_a_p     0x002f       0x001f
        这样在 cudaMemlloc 中才能通过 d_a_p 这个指针来管理 d_a 的地址
    */
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
    add<<<N,1>>>(d_a, d_b, d_c);
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