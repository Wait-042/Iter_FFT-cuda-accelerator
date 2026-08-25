#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>
#include "unwrap_phase_hybrid.cuh"
#include <iostream>
#include <iomanip>
#include <chrono>
#pragma comment(lib, "cufft.lib")

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Rect {
    int y_min, y_max;
    int x_min, x_max;
};

// -------------------------------------------------------------
// 0. 特性萃取 (Traits): 自动适配 float/double 与 CuFFT 复数类型
// -------------------------------------------------------------
template <typename T> struct CuFFTTraits;

template <> struct CuFFTTraits<float> {
    using Complex = cufftComplex;
    static constexpr cufftType C2C_TYPE = CUFFT_C2C;
    static cufftResult execC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) {
        return cufftExecC2C(plan, idata, odata, direction);
    }
};

template <> struct CuFFTTraits<double> {
    using Complex = cufftDoubleComplex;
    static constexpr cufftType C2C_TYPE = CUFFT_Z2Z;
    static cufftResult execC2C(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) {
        return cufftExecZ2Z(plan, idata, odata, direction);
    }
};

// Device 数学函数重载
__device__ __forceinline__ float my_sqrt(float v) { return sqrtf(v); }
__device__ __forceinline__ double my_sqrt(double v) { return sqrt(v); }

__device__ __forceinline__ float my_atan2(float y, float x) { return atan2f(y, x); }
__device__ __forceinline__ double my_atan2(double y, double x) { return atan2(y, x); }

// -------------------------------------------------------------
// 1. CUDA Kernels
// -------------------------------------------------------------

// 后处理 A: 提取 IFFT 后的实部并归一化 (envelope_a)
template <typename T>
__global__ void process_envelope_a_kernel(const typename CuFFTTraits<T>::Complex* in, T* out, int size, T scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx].x * scale;
    }
}

// 后处理 B: 提取 2 * 幅值 (envelope_b) 和 卷绕相位 (phi)
template <typename T>
__global__ void process_envelope_b_and_phi_kernel(const typename CuFFTTraits<T>::Complex* in, T* envelope_b, T* phi, int size, T scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T re = in[idx].x * scale;
        T im = in[idx].y * scale;

        // envelope_b = 2 * abs(val)
        envelope_b[idx] = static_cast<T>(2.0) * my_sqrt(re * re + im * im);

        // phi = angle(val) -> [-pi, pi]
        phi[idx] = my_atan2(im, re);
    }
}

// 2D FFT Shift / IFFT Shift (象限对调)
template <typename T>
__global__ void fftshift_2d_kernel(typename CuFFTTraits<T>::Complex* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width / 2 && y < height / 2) {
        int x2 = x + width / 2;
        int y2 = y + height / 2;

        int idx1 = y * width + x;
        int idx3 = y2 * width + x2;
        typename CuFFTTraits<T>::Complex temp1 = data[idx1];
        data[idx1] = data[idx3];
        data[idx3] = temp1;

        int idx2 = y * width + x2;
        int idx4 = y2 * width + x;
        typename CuFFTTraits<T>::Complex temp2 = data[idx2];
        data[idx2] = data[idx4];
        data[idx4] = temp2;
    }
}

template <typename T>
__global__ void real_to_complex_kernel(const T* d_in, typename CuFFTTraits<T>::Complex* d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx].x = d_in[idx];
        d_out[idx].y = static_cast<T>(0.0);
    }
}

// -------------------------------------------------------------
// 2. 核心 Host 主函数
// -------------------------------------------------------------
template <typename T>
void envelope_phi_by_fft_1d(
    const T* d_img,
    int width,
    int height,
    Rect rect_sel[2],
    T* d_envelope_a,
    T* d_envelope_b,
    T* d_phi
) {
    using ComplexType = typename CuFFTTraits<T>::Complex;

    int size = width * height;
    size_t complex_bytes = size * sizeof(ComplexType);

    // 1. 分配 GPU 内存
    ComplexType* d_img_complex, * d_fft_a, * d_fft_b;
    cudaMalloc(&d_img_complex, complex_bytes);
    cudaMalloc(&d_fft_a, complex_bytes);
    cudaMalloc(&d_fft_b, complex_bytes);

    // 2. 在 GPU 上将实数图像转为复数格式 
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    real_to_complex_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_img, d_img_complex, size);

    // 3. 创建 CuFFT 句柄
    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CuFFTTraits<T>::C2C_TYPE);

    // 4. 正向 2D FFT
    CuFFTTraits<T>::execC2C(plan, d_img_complex, d_img_complex, CUFFT_FORWARD);

    dim3 block2d(16, 16);
    dim3 grid2d_half((width / 2 + block2d.x - 1) / block2d.x, (height / 2 + block2d.y - 1) / block2d.y);

    fftshift_2d_kernel<T> << <grid2d_half, block2d >> > (d_img_complex, width, height);

    // 5. 过滤与 ROI 截取 (使用 2D 内存拷贝降低带宽开销)
    cudaMemset(d_fft_a, 0, complex_bytes);
    cudaMemset(d_fft_b, 0, complex_bytes);

    int h_a = rect_sel[0].y_max - rect_sel[0].y_min;
    int w_a = rect_sel[0].x_max - rect_sel[0].x_min;
    cudaMemcpy2D(
        d_fft_a + rect_sel[0].y_min * width + rect_sel[0].x_min, width * sizeof(ComplexType),
        d_img_complex + rect_sel[0].y_min * width + rect_sel[0].x_min, width * sizeof(ComplexType),
        w_a * sizeof(ComplexType), h_a, cudaMemcpyDeviceToDevice
    );

    int h_b = rect_sel[1].y_max - rect_sel[1].y_min;
    int w_b = rect_sel[1].x_max - rect_sel[1].x_min;
    cudaMemcpy2D(
        d_fft_b + rect_sel[1].y_min * width + rect_sel[1].x_min, width * sizeof(ComplexType),
        d_img_complex + rect_sel[1].y_min * width + rect_sel[1].x_min, width * sizeof(ComplexType),
        w_b * sizeof(ComplexType), h_b, cudaMemcpyDeviceToDevice
    );

    // 6. 逆 FFT (ifftshift -> IFFT)
    fftshift_2d_kernel<T> << <grid2d_half, block2d >> > (d_fft_a, width, height);
    fftshift_2d_kernel<T> << <grid2d_half, block2d >> > (d_fft_b, width, height);

    CuFFTTraits<T>::execC2C(plan, d_fft_a, d_fft_a, CUFFT_INVERSE);
    CuFFTTraits<T>::execC2C(plan, d_fft_b, d_fft_b, CUFFT_INVERSE);

    // 7. 提取包络与包裹相位 Phi (补齐缺失变量定义)
    T fft_scale = static_cast<T>(1.0) / static_cast<T>(width * height);

    process_envelope_a_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_fft_a, d_envelope_a, size, fft_scale);
    process_envelope_b_and_phi_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_fft_b, d_envelope_b, d_phi, size, fft_scale);
    
    // 9. 使用 Herraez 2D 算法（skimage 同款）在 CPU 进行解包裹
    auto start4 = std::chrono::high_resolution_clock::now();

    std::vector<T> h_phi(size);
    std::vector<T> h_phi_unwrap(size);
    cudaMemcpy(h_phi.data(), d_phi, size * sizeof(T), cudaMemcpyDeviceToHost);

    //unwrap_herraez_2d<T>(h_phi.data(), h_phi_unwrap.data(), width, height);
    unwrap_phase_gpu_hybrid<T>(h_phi.data(), h_phi_unwrap.data(), width, height);

    cudaMemcpy(d_phi, h_phi_unwrap.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    auto start5 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(start5 - start4);
    //std::cout << "unwrap_herraez_2d耗时: " << duration4.count() << " ms" << std::endl;

    // 11. 释放 GPU 内存与 Plan
    cufftDestroy(plan);
    cudaFree(d_img_complex);
    cudaFree(d_fft_a);
    cudaFree(d_fft_b);

}

template <typename T>
void envelope_phi_by_fft_2d(
    const T* d_img,
    int width,
    int height,
    Rect rect_sel[3],
    T* d_envelope_a,
    T* d_envelope_bx,
    T* d_envelope_by,
    T* d_phix,
    T* d_phiy
) {

    auto start1 = std::chrono::high_resolution_clock::now();
    using ComplexType = typename CuFFTTraits<T>::Complex;

    int size = width * height;
    size_t complex_bytes = size * sizeof(ComplexType);

    // 1. 分配 GPU 内存
    ComplexType* d_img_complex, * d_fft_a, * d_fft_bx, * d_fft_by;
    cudaMalloc(&d_img_complex, complex_bytes);
    cudaMalloc(&d_fft_a, complex_bytes);
    cudaMalloc(&d_fft_bx, complex_bytes);
    cudaMalloc(&d_fft_by, complex_bytes);

    // 2. 在 GPU 上将实数图像转为复数格式 
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    real_to_complex_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_img, d_img_complex, size);
    auto start2 = std::chrono::high_resolution_clock::now();
    // 3. 创建 CuFFT 句柄
    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CuFFTTraits<T>::C2C_TYPE);

    // 4. 正向 2D FFT
    CuFFTTraits<T>::execC2C(plan, d_img_complex, d_img_complex, CUFFT_FORWARD);
    auto start3 = std::chrono::high_resolution_clock::now();
    dim3 block2d(16, 16);
    dim3 grid2d_half((width / 2 + block2d.x - 1) / block2d.x, (height / 2 + block2d.y - 1) / block2d.y);

    fftshift_2d_kernel<T> << <grid2d_half, block2d >> > (d_img_complex, width, height);
    auto start4 = std::chrono::high_resolution_clock::now();
    // 5. 过滤与 ROI 截取 (使用 2D 内存拷贝降低带宽开销)
    cudaMemset(d_fft_a, 0, complex_bytes);
    cudaMemset(d_fft_bx, 0, complex_bytes);
    cudaMemset(d_fft_by, 0, complex_bytes);

    int h_a = rect_sel[0].y_max - rect_sel[0].y_min;
    int w_a = rect_sel[0].x_max - rect_sel[0].x_min;
    cudaMemcpy2D(
        d_fft_a + rect_sel[0].y_min * width + rect_sel[0].x_min, width * sizeof(ComplexType),
        d_img_complex + rect_sel[0].y_min * width + rect_sel[0].x_min, width * sizeof(ComplexType),
        w_a * sizeof(ComplexType), h_a, cudaMemcpyDeviceToDevice
    );

    int h_bx = rect_sel[1].y_max - rect_sel[1].y_min;
    int w_bx = rect_sel[1].x_max - rect_sel[1].x_min;
    cudaMemcpy2D(
        d_fft_bx + rect_sel[1].y_min * width + rect_sel[1].x_min, width * sizeof(ComplexType),
        d_img_complex + rect_sel[1].y_min * width + rect_sel[1].x_min, width * sizeof(ComplexType),
        w_bx * sizeof(ComplexType), h_bx, cudaMemcpyDeviceToDevice
    );

    int h_by = rect_sel[2].y_max - rect_sel[2].y_min;
    int w_by = rect_sel[2].x_max - rect_sel[2].x_min;
    cudaMemcpy2D(
        d_fft_by + rect_sel[2].y_min * width + rect_sel[2].x_min, width * sizeof(ComplexType),
        d_img_complex + rect_sel[2].y_min * width + rect_sel[2].x_min, width * sizeof(ComplexType),
        w_by * sizeof(ComplexType), h_by, cudaMemcpyDeviceToDevice
    );
    auto start5 = std::chrono::high_resolution_clock::now();
    // 6. 逆 FFT (ifftshift -> IFFT)
    fftshift_2d_kernel<T> << <grid2d_half, block2d >> > (d_fft_a, width, height);
    fftshift_2d_kernel<T> << <grid2d_half, block2d >> > (d_fft_bx, width, height);
    fftshift_2d_kernel<T> << <grid2d_half, block2d >> > (d_fft_by, width, height);
    CuFFTTraits<T>::execC2C(plan, d_fft_a, d_fft_a, CUFFT_INVERSE);
    CuFFTTraits<T>::execC2C(plan, d_fft_bx, d_fft_bx, CUFFT_INVERSE);
    CuFFTTraits<T>::execC2C(plan, d_fft_by, d_fft_by, CUFFT_INVERSE);
    // 7. 提取包络与包裹相位 Phi (补齐缺失变量定义)
    T fft_scale = static_cast<T>(1.0) / static_cast<T>(width * height);

    process_envelope_a_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_fft_a, d_envelope_a, size, fft_scale);
    process_envelope_b_and_phi_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_fft_bx, d_envelope_bx, d_phix, size, fft_scale);
    process_envelope_b_and_phi_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (d_fft_by, d_envelope_by, d_phiy, size, fft_scale);

    // 9. 使用 Herraez 2D 算法（skimage 同款）在 CPU 进行解包裹
    auto start6 = std::chrono::high_resolution_clock::now();

    std::vector<T> h_phix(size), h_phiy(size);
    std::vector<T> h_phix_unwrap(size), h_phiy_unwrap(size);
    cudaMemcpy(h_phix.data(), d_phix, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phiy.data(), d_phiy, size * sizeof(T), cudaMemcpyDeviceToHost);
    auto start7 = std::chrono::high_resolution_clock::now();

    //unwrap_herraez_2d<T>(h_phix.data(), h_phix_unwrap.data(), width, height);
    //unwrap_herraez_2d<T>(h_phiy.data(), h_phiy_unwrap.data(), width, height);

    unwrap_phase_gpu_hybrid<T>(h_phix.data(), h_phix_unwrap.data(), width, height);
    unwrap_phase_gpu_hybrid<T>(h_phiy.data(), h_phiy_unwrap.data(), width, height);
    auto start8 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_phix, h_phix_unwrap.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phiy, h_phiy_unwrap.data(), size * sizeof(T), cudaMemcpyHostToDevice);

    auto start9 = std::chrono::high_resolution_clock::now();

    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start1);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(start3 - start2);
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(start4 - start3);
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(start5 - start4);
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(start6 - start5);
    auto duration6 = std::chrono::duration_cast<std::chrono::milliseconds>(start7 - start6);
    auto duration7 = std::chrono::duration_cast<std::chrono::milliseconds>(start8 - start7);
    auto duration8 = std::chrono::duration_cast<std::chrono::milliseconds>(start9 - start8);
    auto duration9 = std::chrono::duration_cast<std::chrono::milliseconds>(start9 - start1);

    //std::cout << "内存分配耗时: " << duration1.count() << " ms" << std::endl;
    //std::cout << "FFT耗时: " << duration2.count() << " ms" << std::endl;
    //std::cout << "FFTSHIFT耗时: " << duration3.count() << " ms" << std::endl;
    //std::cout << "频谱截取耗时: " << duration4.count() << " ms" << std::endl;
    //std::cout << "逆变换耗时: " << duration5.count() << " ms" << std::endl;
    //std::cout << "phi gpu->cpu耗时: " << duration6.count() << " ms" << std::endl;
    //std::cout << "相位解包裹耗时: " << duration7.count() << " ms" << std::endl;
    //std::cout << "phi cpu->gpu耗时: " << duration8.count() << " ms" << std::endl;
    //std::cout << "整体程序耗时: " << duration9.count() << " ms" << std::endl;

    // 11. 释放 GPU 内存与 Plan
    cufftDestroy(plan);
    cudaFree(d_img_complex);
    cudaFree(d_fft_a);
    cudaFree(d_fft_bx);
    cudaFree(d_fft_by);

}