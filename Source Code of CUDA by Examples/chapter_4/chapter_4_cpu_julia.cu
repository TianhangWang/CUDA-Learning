#include "cpu_bitmap.h"
#include <iostream>
#include <chrono>
#define DIM 512

class cuComplex
{
public:
    cuComplex(float r=0, double i=0)
        : re(r), im(i)
    {}    
    float magnitude2(void) {return re*re + im*im;}
    cuComplex operator*(const cuComplex& a){
        return cuComplex(this->re * a.re - this->im * a.im, 
                         this->im * a.re + this->re * a.im);
    }
    cuComplex operator+(const cuComplex& a){
        return cuComplex(this->re + a.re, this->im + a.im);
    }

private:
    float re, im;
};

int julia(int x, int y){

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

void kernel(unsigned char *ptr){
    for (int y=0; y< DIM; y++){
        for (int x=0; x< DIM; x++){
            int offset = x + y * DIM;
            /*
            这里需要注意 bitmap 的结构与存储形式;
            bitmap 本身是一个 (DIM, DIM, 4): height = DIM; Width = DIM; channel = 4;
            而存储的指针却是指向一维数组，所以需要将channel展平; eg.
            [[0,0,0],[0,0,1],[0,0,2],[0,0,3],[1,0,0],...,]
            其中[0,0,0] 前面两个表示对应的(x,y)，第三个是第一个通道的索引;
            */
           int juliaValue = julia(x, y);
           ptr[offset*4 + 0] = 0;
           ptr[offset*4 + 1] = 255 * juliaValue;
           ptr[offset*4 + 2] = 0;
           ptr[offset*4 + 3] = 255;
        }
    }
}

int main(void){
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();
    auto start = std::chrono::steady_clock::now();
    kernel(ptr);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    std::cout << "The run time is: " << elapsed.count() << "us" << std::endl;
    
    bitmap.display_and_exit();
}
