# Chapter 3

1. p26; 需要注意的点是为 device 声明的指针a内的*a值在 host 上是不可以读写的！因为CUDA只是利用了 host 上声明的&a地址，里面的值是unknown的，随意读写可能造成程序崩溃；在此基础上，也不难理解，为什么释放内存要用 cudafree(), 不用 free(), 因为内容根本就不在 host 为 device 声明的指针a地址的值中;
    ```
                    地址          值
    Host:
        d_a       0x001f      unknown  -> host 上为 device 声明的地址指针
        d_a_p     0x002f       0x001f  -> cudaMolloc() 用于管理地址指针的指针
    ```
2. p30-32; 简单介绍了一下如何查询设备的情况，这可以直接用 `deviceQuery.exe` 来查看，不一定要用程序去看 `cudaGetDeviceCount(), cudaGetDeviceProperties()`; 查这个属性一般是为了例如显卡是否支持双精度计算？

3. p34; 基于一些设定去选择我们所需要显卡; 
    ```c
    cudaDeviceProp  prop;
    int dev;
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    // 设定你需要的性能
    prop.major = 1;
    prop.minor = 3;
    // 挑一下显卡
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
    // 设定显卡
    HANDLE_ERROR( cudaSetDevice( dev ) );
    ```
# Chapter 4

1. p44; 这里主要将之前在 `VectorAddition_block.cu` 代码中的关键字 `blockIdx` 进行了更为深刻的理解; 
    a. 为什么 `blockIdx` 不需要定义，就可以使用？
        i. 因为这是 CUDA builit-in 变量，在运行时 CUDA 会给我们定义的;
    b. 为什么有时候 `blockIdx.x` 有 `.x`?
        i. 因为 cuda 原始是为了方便图像处理或者矩阵计算，而图像索引自然有(x,y)两维; 在我们的问题中，使用一维的 x 就已经足够，所以`.x` 并不奇怪，甚至可以统一换为 `.y`;

2. p47; JULIA SET 代码; 详情见源码; 这里注意的是，在 CPU 调用模式下，耗时为 132686 us; 而采用 GPU 加速之后，耗时为 27.4 us; 这样带来的性能提升可达到 4842 倍！
    <p align="center">
    <img src="figures/julia.png" width="40%">
    </p>


