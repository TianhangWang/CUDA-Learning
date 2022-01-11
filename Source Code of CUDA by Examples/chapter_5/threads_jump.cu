#include <iostream>
#define N (33 * 1024)

__global__ void add(int *a, int *b, int *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N){
        c[index] = a[index] + b[index];
        index += blockDim.x * gridDim.x;
    }
    /*
    统一解释一下
    blockDim.x,y,z 表示在某个特定方向上，一个block有多少个线程数 threads;
    gridDim.x,y,z  表示在某个特定方向上，整个计算图上有多少个 blocks;
    blockDim.x * gridDim.x 表示在 x 方向上，整个计算图所有的线程数 threads;

    所以针对 Line6-line9的代码其实是表示，在计算长度为blockDim.x*gridDim.x长度的内容后，
    继续向后顺延同样长度；
   first time         second time             
index  | ----- .... -----| | ----- .... -----|
input  [                                           ]    
    作用：
        对应很长的输入，gpu所调度的计算单元有限，所以需要重复调用这些计算单元;
    */
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
    add<<<128,128>>>(d_a, d_b, d_c);
    // copy result back
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // check for correct
    bool success = true;
    for (int i=0; i<N; i++){
        if ((a[i] + b[i]) != c[i]){
            std::cout << "Error: " << a[i] << "+" << b[i] << "!=" << c[i] << '\n';
            success=false;
        }
    }
    if (success) {
        std::cout << "Succeed!" << '\n';
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