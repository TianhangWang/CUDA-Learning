#include <iostream>

int main(){
    int count;
    cudaGetDeviceCount(&count);
    std::cout << count << std::endl;

    return 0;
}