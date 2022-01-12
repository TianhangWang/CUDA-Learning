#include <iostream>
#include <stdlib.h>
#include "../depend/cpu_bitmap.h"
#define INF 2e10f
#define SPHERES 50
#define DIM  512
#define rnd( x ) (x * rand() / RAND_MAX)

struct Sphere {
    float r,b,g;
    float radius;
    float x,y,z; // 球心坐标

    __device__ float hit (const float& ox, const float& oy, float *n){
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius * radius){
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            *n = dz / radius;   // 计算光线切角，用于设置球谁在前[亮]，谁在后[暗]
            return z-dz; // 给外面用于判断是否属于光追
        }
        return -INF;
    }

};

__global__ void kernel(unsigned char *ptr, Sphere *s){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;
    // 可以对比 chapter_5/threads_jump.cu; 区别也就是计算单元与原输入单元的关系，是不是线程数
    // 就是等于输入单元数，这时候就不用jump; 如果线程数是小于输入单元的，就需要重复利用了

    float ox = (x - DIM/2);
    float oy = (y - DIM/2);

    float r=0, g=0, b=0;
    float maxz = -INF;
    float minz = INF;

    for(int i=0; i<SPHERES; i++){
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz){ // 决定要画
            if (t < minz){ // 画哪一部分
                float fscale = n;
                r = s[i].r * fscale;
                g = s[i].g * fscale;
                b = s[i].b * fscale;
                minz = t;
            }
        }
    }
    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;    
}

int main(){
    Sphere *s;
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    cudaMalloc((void **)&dev_bitmap, bitmap.image_size());
    cudaMalloc((void **)&s, sizeof(Sphere) * SPHERES);

    Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);

    // 初始化空间球信息
    for (int i=0; i<SPHERES; i++){
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f);
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16,16);

    kernel<<<grids, threads>>>(dev_bitmap, s);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);
    cudaFree(s);
    free(temp_s);

}
