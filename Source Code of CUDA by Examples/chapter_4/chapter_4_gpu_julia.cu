#include "../depend/cpu_bitmap.h"
#include <iostream>
#include <chrono>
#define DIM 512

class cuComplex
{
public:
    __device__ cuComplex(float r=0, double i=0)
        : re(r), im(i)
    {}    
    __device__ float magnitude2(void) {return re*re + im*im;}
    __device__ cuComplex operator*(const cuComplex& a){
        return cuComplex(this->re * a.re - this->im * a.im, 
                         this->im * a.re + this->re * a.im);
    }
    __device__ cuComplex operator+(const cuComplex& a){
        return cuComplex(this->re + a.re, this->im + a.im);
    }

private:
    float re, im;
};

__device__ int julia(int x, int y){

    const float scale = 1.5;
    float jx = scale * (float) (DIM/2 - x)/(DIM/2);
    float jy = scale * (float) (DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++){
        a = a * a + c;
        if (a.magnitude2() > 1000){
            return 0;
        }
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr){

    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    // 这里就是给 block 划分 index ；
    // x 属于 [0-DIM], y 属于[0-DIM]; gridDim = DIM; 结合之前说到的一维数组指针的存储方法，即可理解
    int juliaValue = julia(x, y);
    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = 255 * juliaValue;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

int main(void){
    

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    cudaMalloc((void **)&dev_bitmap, bitmap.image_size());
    dim3 grid(DIM, DIM); // 其实是(DIM,DIM,1) 所以用dim3

    auto start = std::chrono::steady_clock::now();
    kernel<<<grid,1>>>(dev_bitmap);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    std::cout << "The run time is: " << elapsed.count() << "us" << std::endl;
    
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);

}
