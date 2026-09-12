# Iter_FFT-cuda-accelerator

## 介绍
一个使用 CUDA 和 C++ 加速的迭代FFT提算法，用于提取二维条纹图像的相位。相比 Python numpy版本大约快15倍，比Python cupy版本快大约9倍。

## 环境:
```
Graphics Card: RTX 4070 Super

Python 3.11

CUDA Version 13.2

CUDAToolkit 12.8

g++(Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

## quick start
我们提供了算法的 Python NumPy 版本和 CuPy 版本，你可以通过 config.py 中的 'CUPY' 参数在它们之间切换。

### Python
```
cd Iter_FFT_py
pip install -r requirements.txt
python iter_fft.py
```

### CUDA
这是个优化版本，相比于旧版本做了内存分配优化以及一些细节调优
```
cd Iter_FFT_opt_cu
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./app
```

这是个旧版本
```
cd Iter_FFT_cu
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./app
```

## result

Python 版本和 CUDA 版本的结果差异：
![Python_and_CUDA_result_diff](Iter_FFT_py/Python_and_CUDA_result_diff.png)

同时，使用迭代 FFT 算法和普通 FFT 算法提取的包络相位与理想包络相位的差异:
![diff_between_the_results_FFT_and_Iter_FFT_and_the_ideal_values.png](Iter_FFT_py/diff_between_the_results_FFT_and_Iter_FFT_and_the_ideal_values.png)


核心模块耗时对比：

| 核心耗时模块           | numpy    | cupy     | CUDA&C++  |
|------------------|----------|----------|-----------|
| zernike基生成 / s   | 2.1      | 0.08     | 0.01      |
| zernike基伪逆生成 / s | 2        | 0.22     | 0.1       |
| FFT变换 / s        | 0.4 * 7  | 0.02 * 7 | 0.01 * 7  |
| zernike低通滤波 / s  | 1.25 * 6 | 0.08 * 6 | 0.03 * 6  |
| 相位解包裹 / s        | 1.4 * 14 | 1.4 * 14 | 0.11 * 14 |
| 其他 / s           | 0.4      | 0.02     | 0.43      |
| 整体耗时 / s         | 34.4     | 20.54    | 2.33      |

PS：表中 a * b 中a表示的是单次耗时，b是执行次数 

这里注意到，相位展开的时间对于 numpy 和 cupy 来说是一样的，因为相位展开是一个高度串行的算法。
在 Python 中，它调用了第三方库 skimage.restoration.unwrap_phase，而这个库不支持 GPU 加速。
他是我们算法的核心耗时瓶颈，因为算法核心是一个高度串行化的路径寻优并查集算法

在 CUDA 版本中，一开始我们尝试了用一个单线程去实现，效果很差，运行完要几十s，后面尝试了一种基于有限差分的快速傅里叶变换与离散余弦变换相位解包裹算法，
虽然该算法使得运行时间得到了明显降低，但是算法会把某些误差分散到全局，使得整体相位求解精度不够。

后来我们阅读了 C 源码，识别到整个算法中除并查集外的部分如计算可靠性、排序、路径展开压缩和偏移量写回是适合做并行优化的，于是我们对这些模块进行了 CUDA 优化，
同时应用字节对齐、数据预取等手段优化数据访问，最终通过CPU+GPU混合解包裹流程将相位解包裹加速大约7倍

同时，代码整体选用double类型是因为项目追求高精度的相位结果，使用float后精度会下降几个数量级，当然，当前算法耗时的瓶颈并不在数据类型上

对于整个算法，我们的 CUDA 版本相比 numpy 版本快了将近 15 倍，相比 numpy 版本快了将近 9 倍，但是相位解包裹仍然是一个相对耗时的操作，
因为强数据依赖导致GPU资源没有得到充分利用，这需要进一步研究并行化相位解包裹的方案

## 参考

- [NVIDIA Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#)
- [The User Guide for Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)

## 下一步计划

- 考虑其他方式来并行化相位展开，或者可能使用分块展开，使用16*16的块分割图像，每个块做相位解包裹，之后块间偏移量做修正，类似并行规约的思想（不过这现在只是个想法）
- 选取ROI区域，减少解包裹数据量