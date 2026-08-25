#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

#include "system_error_generate.cuh"
#include "zernike_fitting.cuh"
#include "frequency_analysis.cuh"


#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define M_PI 3.14159265358979323846

// 错误检查宏
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << err << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUSOLVER_CHECK(err) \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER Error: " << err << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

template <typename T>
void save_raw(const std::string& filename, const std::vector<T>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        size_t total_bytes = data.size() * sizeof(T);
        file.write(reinterpret_cast<const char*>(data.data()), total_bytes);
        file.close();
        std::cout << "成功保存到 " << filename
            << " (大小: " << total_bytes / (1024.0 * 1024.0) << " MB)" << std::endl;
    }
    else {
        std::cerr << "错误：无法打开文件 " << filename << " 进行写入！" << std::endl;
    }

}

template <typename T>
__global__ void coordinate_generate_kernel(int M, T norm_val, T step, T* rho, T* theta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= M) return;

    T x_norm = static_cast<T>(-norm_val + col * step);
    T y_norm = static_cast<T>(-norm_val + row * step);

    rho[row * M + col] = static_cast<T>(sqrt(x_norm * x_norm + y_norm * y_norm));
    theta[row * M + col] = static_cast<T>(atan2(y_norm, x_norm));

}

template <typename T>
__global__ void img_generate(T* d_img, T* d_rho, T* d_a, T* d_b, T* d_phi, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * M + col;

    if (row >= M || col >= M) return;
    if (d_rho[idx] < 1.0) {
        d_img[idx] = d_a[idx] + d_b[idx] * cos(d_phi[idx]);
    }
    else {
        d_img[idx] = 0.0;
    }
    
}

template <typename T>
__global__ void img_generate_2d(T* d_img, T* d_rho, T* d_a, T* d_bx, T* d_by, T* d_phix, T* d_phiy, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * M + col;

    if (row >= M || col >= M) return;
    if (d_rho[idx] < 1.0) {
        d_img[idx] = d_a[idx] + d_bx[idx] * cos(d_phix[idx]) + d_by[idx] * cos(d_phiy[idx]);
    }
    else {
        d_img[idx] = 0.0;
    }

}

template <typename T>
__global__ void fill_kernel(T* ptr, T val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = val;
    }
}

template <typename T>
__global__ void subtract_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

template <typename T>
void iter_fft_1d(Rect rect_sel[2], T* d_img, T* d_pinv, T* d_zernike_basis, T* d_rho, T rho_range_fit, int* d_mask_indices, int M, int n_order, int iter_num, int* h_M, int start_mode, int end_mode,
    T* d_a_corr, T* d_b_corr, T* d_phi_corr) {

    T* d_a_fft, * d_b_fft, * d_phi_fft;
    int total_pixes = M * M;

    cudaMalloc(&d_a_fft, total_pixes * sizeof(T));
    cudaMalloc(&d_b_fft, total_pixes * sizeof(T));
    cudaMalloc(&d_phi_fft, total_pixes * sizeof(T));
    envelope_phi_by_fft_1d<T>(d_img, M, M, rect_sel, d_a_fft, d_b_fft, d_phi_fft);

    T* d_a_model, * d_b_model, * d_phi_model;
    cudaMalloc((void**)&d_a_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_b_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_phi_model, total_pixes * sizeof(T));
    zernike_low_pass_filter<T>(d_a_fft, d_a_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
    zernike_low_pass_filter<T>(d_b_fft, d_b_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
    zernike_low_pass_filter<T>(d_phi_fft, d_phi_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);

    dim3 cor_threads(16, 16);
    dim3 cor_blocks(CEIL(M, 16), CEIL(M, 16));
    img_generate<T> << <cor_blocks, cor_threads >> > (d_img, d_rho, d_a_model, d_b_model, d_phi_model, M);

    T* d_a_fft_model, * d_b_fft_model, * d_phi_fft_model;
    cudaMalloc((void**)&d_a_fft_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_b_fft_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_phi_fft_model, total_pixes * sizeof(T));
    envelope_phi_by_fft_1d<T>(d_img, M, M, rect_sel, d_a_fft_model, d_b_fft_model, d_phi_fft_model);

    T* d_delta_a, * d_delta_b, * d_delta_phi;
    cudaMalloc((void**)&d_delta_a, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_delta_b, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_delta_phi, total_pixes * sizeof(T));
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft_model, d_a_model, d_delta_a, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_b_fft_model, d_b_model, d_delta_b, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phi_fft_model, d_phi_model, d_delta_phi, total_pixes);

    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft, d_delta_a, d_a_corr, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_b_fft, d_delta_b, d_b_corr, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phi_fft, d_delta_phi, d_phi_corr, total_pixes);

    for (int i = 0; i < iter_num; i++) {
        zernike_low_pass_filter<T>(d_a_corr, d_a_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
        zernike_low_pass_filter<T>(d_b_corr, d_b_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
        zernike_low_pass_filter<T>(d_phi_corr, d_phi_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);

        img_generate<T> << <cor_blocks, cor_threads >> > (d_img, d_rho, d_a_model, d_b_model, d_phi_model, M);

        envelope_phi_by_fft_1d<T>(d_img, M, M, rect_sel, d_a_fft_model, d_b_fft_model, d_phi_fft_model);

        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft_model, d_a_model, d_delta_a, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_b_fft_model, d_b_model, d_delta_b, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phi_fft_model, d_phi_model, d_delta_phi, total_pixes);

        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft, d_delta_a, d_a_corr, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_b_fft, d_delta_b, d_b_corr, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phi_fft, d_delta_phi, d_phi_corr, total_pixes);
    }

    cudaFree(d_a_fft);
    cudaFree(d_b_fft);
    cudaFree(d_phi_fft);
    cudaFree(d_a_model);
    cudaFree(d_b_model);
    cudaFree(d_phi_model);
    cudaFree(d_a_fft_model);
    cudaFree(d_b_fft_model);
    cudaFree(d_phi_fft_model);

    cudaFree(d_delta_a);
    cudaFree(d_delta_b);
    cudaFree(d_delta_phi);

}

template <typename T>
void iter_fft_2d(Rect rect_sel[2], T* d_img, T* d_pinv, T* d_zernike_basis, T* d_rho, T rho_range_fit, int* d_mask_indices, int M, int n_order, int iter_num, int* h_M, int start_mode, int end_mode,
    T* d_a_corr, T* d_bx_corr, T* d_by_corr, T* d_phix_corr, T* d_phiy_corr) {

    auto start1 = std::chrono::high_resolution_clock::now();

    int total_pixes = M * M;

    T* d_a_fft, * d_bx_fft, * d_by_fft, * d_phix_fft, * d_phiy_fft;
    cudaMalloc(&d_a_fft, total_pixes * sizeof(T));
    cudaMalloc(&d_bx_fft, total_pixes * sizeof(T));
    cudaMalloc(&d_by_fft, total_pixes * sizeof(T));
    cudaMalloc(&d_phix_fft, total_pixes * sizeof(T));
    cudaMalloc(&d_phiy_fft, total_pixes * sizeof(T));

    T* d_a_model, * d_bx_model, * d_by_model, * d_phix_model, * d_phiy_model;
    cudaMalloc((void**)&d_a_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_bx_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_by_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_phix_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_phiy_model, total_pixes * sizeof(T));

    T* d_a_fft_model, * d_bx_fft_model, * d_by_fft_model, * d_phix_fft_model, * d_phiy_fft_model;
    cudaMalloc((void**)&d_a_fft_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_bx_fft_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_by_fft_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_phix_fft_model, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_phiy_fft_model, total_pixes * sizeof(T));

    T* d_delta_a, * d_delta_bx, * d_delta_by, * d_delta_phix, * d_delta_phiy;
    cudaMalloc((void**)&d_delta_a, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_delta_bx, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_delta_by, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_delta_phix, total_pixes * sizeof(T));
    cudaMalloc((void**)&d_delta_phiy, total_pixes * sizeof(T));

    auto start2 = std::chrono::high_resolution_clock::now();

    envelope_phi_by_fft_2d<T>(d_img, M, M, rect_sel, d_a_fft, d_bx_fft, d_by_fft, d_phix_fft, d_phiy_fft);

    auto start3 = std::chrono::high_resolution_clock::now();

    zernike_low_pass_filter<T>(d_a_fft, d_a_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
    zernike_low_pass_filter<T>(d_bx_fft, d_bx_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
    zernike_low_pass_filter<T>(d_by_fft, d_by_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
    zernike_low_pass_filter<T>(d_phix_fft, d_phix_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
    zernike_low_pass_filter<T>(d_phiy_fft, d_phiy_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
    auto start4 = std::chrono::high_resolution_clock::now();

    dim3 cor_threads(16, 16);
    dim3 cor_blocks(CEIL(M, 16), CEIL(M, 16));
    img_generate_2d<T> << <cor_blocks, cor_threads >> > (d_img, d_rho, d_a_model, d_bx_model, d_by_model, d_phix_model, d_phiy_model, M);
    auto start5 = std::chrono::high_resolution_clock::now();

    envelope_phi_by_fft_2d<T>(d_img, M, M, rect_sel, d_a_fft_model, d_bx_fft_model, d_by_fft_model, d_phix_fft_model, d_phiy_fft_model);
    auto start6 = std::chrono::high_resolution_clock::now();

    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft_model, d_a_model, d_delta_a, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_bx_fft_model, d_bx_model, d_delta_bx, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_by_fft_model, d_by_model, d_delta_by, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phix_fft_model, d_phix_model, d_delta_phix, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phiy_fft_model, d_phiy_model, d_delta_phiy, total_pixes);

    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft, d_delta_a, d_a_corr, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_bx_fft, d_delta_bx, d_bx_corr, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_by_fft, d_delta_by, d_by_corr, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phix_fft, d_delta_phix, d_phix_corr, total_pixes);
    subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phiy_fft, d_delta_phiy, d_phiy_corr, total_pixes);
    auto start7 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iter_num; i++) {
        zernike_low_pass_filter<T>(d_a_corr, d_a_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
        zernike_low_pass_filter<T>(d_bx_corr, d_bx_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
        zernike_low_pass_filter<T>(d_by_corr, d_by_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
        zernike_low_pass_filter<T>(d_phix_corr, d_phix_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);
        zernike_low_pass_filter<T>(d_phiy_corr, d_phiy_model, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, M, n_order, h_M, start_mode, end_mode);

        img_generate_2d<T> << <cor_blocks, cor_threads >> > (d_img, d_rho, d_a_model, d_bx_model, d_by_model, d_phix_model, d_phiy_model, M);

        envelope_phi_by_fft_2d<T>(d_img, M, M, rect_sel, d_a_fft_model, d_bx_fft_model, d_by_fft_model, d_phix_fft_model, d_phiy_fft_model);

        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft_model, d_a_model, d_delta_a, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_bx_fft_model, d_bx_model, d_delta_bx, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_by_fft_model, d_by_model, d_delta_by, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phix_fft_model, d_phix_model, d_delta_phix, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phiy_fft_model, d_phiy_model, d_delta_phiy, total_pixes);

        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_a_fft, d_delta_a, d_a_corr, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_bx_fft, d_delta_bx, d_bx_corr, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_by_fft, d_delta_by, d_by_corr, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phix_fft, d_delta_phix, d_phix_corr, total_pixes);
        subtract_kernel<T> << <CEIL(total_pixes, 512), 512 >> > (d_phiy_fft, d_delta_phiy, d_phiy_corr, total_pixes);
    }
    auto start8 = std::chrono::high_resolution_clock::now();

    cudaFree(d_a_fft);
    cudaFree(d_bx_fft);
    cudaFree(d_by_fft);
    cudaFree(d_phix_fft);
    cudaFree(d_phiy_fft);
    cudaFree(d_a_model);
    cudaFree(d_bx_model);
    cudaFree(d_by_model);
    cudaFree(d_phix_model);
    cudaFree(d_phiy_model);
    cudaFree(d_a_fft_model);
    cudaFree(d_bx_fft_model);
    cudaFree(d_by_fft_model);
    cudaFree(d_phix_fft_model);
    cudaFree(d_phiy_fft_model);

    cudaFree(d_delta_a);
    cudaFree(d_delta_bx);
    cudaFree(d_delta_by);
    cudaFree(d_delta_phix);
    cudaFree(d_delta_phiy);
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
    //std::cout << "envelope_phi_by_fft_2d耗时: " << duration2.count() << " ms" << std::endl;
    //std::cout << "zernike低通滤波耗时: " << duration3.count() << " ms" << std::endl;
    //std::cout << "图像恢复耗时: " << duration4.count() << " ms" << std::endl;
    //std::cout << "envelope_phi_by_fft_2d耗时: " << duration5.count() << " ms" << std::endl;
    //std::cout << "修正耗时: " << duration6.count() << " ms" << std::endl;
    //std::cout << "迭代修正耗时: " << duration7.count() << " ms" << std::endl;
    //std::cout << "内存释放耗时: " << duration8.count() << " ms" << std::endl;
    //std::cout << "整体程序耗时: " << duration9.count() << " ms" << std::endl;

}

template <typename T>
void coordinate_generate_kernel_test(void) {
    using Real = T;
    int M = 2048;
    int img_pixes = M * M;

    const Real spot_radius_nm = 9e6;
    const Real pixel_size_nm = 10e3;
    Real cmos_radius = M * pixel_size_nm / 2.0;

    Real* d_rho, * d_theta;
    cudaMalloc((void**)&d_rho, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_theta, img_pixes * sizeof(Real));

    ////////////////////////// rho theta generate ///////////////////////////
    dim3 cor_threads(16, 16);
    dim3 cor_blocks(CEIL(M, 16), CEIL(M, 16));
    Real norm_val = static_cast<Real>(cmos_radius / spot_radius_nm);
    Real step = static_cast<Real>(2.0) * norm_val / (M - 1);
    coordinate_generate_kernel<Real> << <cor_blocks, cor_threads >> > (M, norm_val, step, d_rho, d_theta);

    cudaFree(d_rho);
    cudaFree(d_theta);
}


int main(void) {
    auto start1 = std::chrono::high_resolution_clock::now();

    using Real = double;
    //using Real = float;

    int M = 2048;
    int img_pixes = M * M;
    int N_ORDER = 64;
    bool is_norm = false;
    int iter_num = 5;
    bool mix_precision = true;

    const Real spot_radius_nm = 9e6;
    const Real pixel_size_nm = 10e3;
    Real cmos_radius = M * pixel_size_nm / 2.0;
    Real rho_range_fit = 0.995;
    Real ddx = 27e3;
    Real ddy = 27e3;
    Real ddz = 9e6 / std::tan(std::asin(1.35 / 4.0));
    Real scale_factor = 2 * M_PI / 193.368;

    std::vector<Real> zernike_basis(N_ORDER * img_pixes);

    Real* d_rho, * d_theta, * d_zernike_basis;
    cudaMalloc((void**)&d_rho, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_theta, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_zernike_basis, N_ORDER * img_pixes * sizeof(Real));

    Real* d_phi_x, * d_phi_y;
    cudaMalloc((void**)&d_phi_x, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_phi_y, img_pixes * sizeof(Real));

    Real* d_img_2d, * d_a, * d_b;
    cudaMalloc(&d_img_2d, img_pixes * sizeof(Real));
    cudaMalloc(&d_a, img_pixes * sizeof(Real));
    cudaMalloc(&d_b, img_pixes * sizeof(Real));

    Real* d_aa_corr, * d_bx_corr, * d_by_corr, * d_phix_corr, * d_phiy_corr;
    cudaMalloc((void**)&d_aa_corr, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_bx_corr, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_by_corr, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_phix_corr, img_pixes * sizeof(Real));
    cudaMalloc((void**)&d_phiy_corr, img_pixes * sizeof(Real));

    auto start2 = std::chrono::high_resolution_clock::now();

    ////////////////////////// rho theta generate ///////////////////////////
    dim3 cor_threads(16, 16);
    dim3 cor_blocks(CEIL(M, 16), CEIL(M, 16));
    Real norm_val = static_cast<Real>(cmos_radius / spot_radius_nm);
    Real step = static_cast<Real>(2.0) * norm_val / (M - 1);
    coordinate_generate_kernel<Real> << <cor_blocks, cor_threads >> > (M, norm_val, step, d_rho, d_theta);

    ////////////////////////// sys_phi ///////////////////////////
    dim3 sys_threads(16, 16);
    dim3 sys_blocks(CEIL(M, 16), CEIL(M, 16));

    system_error_generate<Real>(cmos_radius, ddx, ddy, ddz, scale_factor, d_phi_x, d_phi_y, M);

    ////////////////////////// img generate ///////////////////////////
    fill_kernel<Real> << <CEIL(img_pixes, 512), 512 >> > (d_a, static_cast<Real>(1.0), img_pixes);
    fill_kernel<Real> << <CEIL(img_pixes, 512), 512 >> > (d_b, static_cast<Real>(1.0), img_pixes);
    img_generate_2d<Real> << <cor_blocks, cor_threads >> > (d_img_2d, d_rho, d_a, d_b, d_b, d_phi_x, d_phi_y, M);

    auto start3 = std::chrono::high_resolution_clock::now();

    ////////////////////////// zernike_basis_generate ///////////////////////////
    if (mix_precision) {
        generate_zernike_basis_optimized<Real, true>(d_rho, d_theta, d_zernike_basis, M, M, N_ORDER, is_norm);
    }
    else {
        generate_zernike_basis_optimized<Real, false>(d_rho, d_theta, d_zernike_basis, M, M, N_ORDER, is_norm);
    }

    ////////////////////////// zernike_basis_pinv ///////////////////////////
    Real* d_pinv = nullptr;
    int h_M = 0;
    int* d_mask_indices = nullptr;
    generate_mask<Real>(d_mask_indices, d_rho, rho_range_fit, M, M, &h_M);
    zerk_fit_pinv_generate_cuda<Real>(d_zernike_basis,
        d_rho, 
        d_mask_indices,
        N_ORDER, M, M,
        &d_pinv,
        &h_M);
    
    auto start4 = std::chrono::high_resolution_clock::now();

    ////////////////////////// iter_fft ///////////////////////////
    int start_mode = 0;
    int end_mode = 64;
    Rect rect_sel[3];
    // 区域 A (低频背景区)
    rect_sel[0] = { 974, 1074, 974, 1074 };
    // 区域 B (单侧频偏信号区，用于提包络和相位)
    rect_sel[1] = { 974, 1074, 1074, 1174 };
    rect_sel[2] = { 1074, 1174, 974, 1074 };

    iter_fft_2d<Real>(rect_sel, d_img_2d, d_pinv, d_zernike_basis, d_rho, rho_range_fit, d_mask_indices, M, N_ORDER, iter_num, &h_M, start_mode, end_mode, d_aa_corr, d_bx_corr, d_by_corr, d_phix_corr, d_phiy_corr);
    //std::vector<Real> aa_corr(img_pixes);
    //std::vector<Real> bx_corr(img_pixes);
    //std::vector<Real> by_corr(img_pixes);
    //std::vector<Real> phix_corr(img_pixes);
    //std::vector<Real> phiy_corr(img_pixes);
    //cudaMemcpy(aa_corr.data(), d_aa_corr, img_pixes * sizeof(Real), cudaMemcpyDeviceToHost);
    //cudaMemcpy(bx_corr.data(), d_bx_corr, img_pixes * sizeof(Real), cudaMemcpyDeviceToHost);
    //cudaMemcpy(by_corr.data(), d_by_corr, img_pixes * sizeof(Real), cudaMemcpyDeviceToHost);
    //cudaMemcpy(phix_corr.data(), d_phix_corr, img_pixes * sizeof(Real), cudaMemcpyDeviceToHost);
    //cudaMemcpy(phiy_corr.data(), d_phiy_corr, img_pixes * sizeof(Real), cudaMemcpyDeviceToHost);
    //save_raw<Real>("aa_corr_float64.bin", aa_corr);
    //save_raw<Real>("bx_corr_float64.bin", bx_corr);
    //save_raw<Real>("by_corr_float64.bin", by_corr);
    //save_raw<Real>("phix_corr_float64.bin", phix_corr);
    //save_raw<Real>("phiy_corr_float64.bin", phiy_corr);
    auto start5 = std::chrono::high_resolution_clock::now();

    cudaFree(d_img_2d);
    cudaFree(d_aa_corr);
    cudaFree(d_bx_corr);
    cudaFree(d_by_corr);
    cudaFree(d_phix_corr);
    cudaFree(d_phiy_corr);

    cudaFree(d_rho);
    cudaFree(d_theta);
    cudaFree(d_zernike_basis);
    cudaFree(d_mask_indices);
    cudaFree(d_pinv);
    cudaFree(d_phi_x);
    cudaFree(d_phi_y);
    cudaFree(d_a);
    cudaFree(d_b);

    auto start6 = std::chrono::high_resolution_clock::now();

    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start1);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(start3 - start2);
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(start4 - start3);
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(start5 - start4);
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(start6 - start5);
    auto duration6= std::chrono::duration_cast<std::chrono::milliseconds>(start6 - start1);

    std::cout << "内存分配耗时: " << duration1.count() << " ms" << std::endl;
    std::cout << "相位、图像生成耗时: " << duration2.count() << " ms" << std::endl;
    std::cout << "zernike基生成耗时: " << duration3.count() << " ms" << std::endl;
    std::cout << "2d条纹迭代FFT提取相位耗时: " << duration4.count() << " ms" << std::endl;
    std::cout << "内存释放耗时: " << duration5.count() << " ms" << std::endl;
    std::cout << "整体程序耗时: " << duration6.count() << " ms" << std::endl;

    return 0;
}