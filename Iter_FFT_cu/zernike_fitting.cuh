#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cusolver.lib")

#define MAX_ZERNIKE_COEFFS 10 // 最多支持到 n = 18 阶

// ==========================================
// 1. cuBLAS / cuSOLVER 模板 Wrapper 封装
// ==========================================
template <typename T>
cublasStatus_t cublasGemv_wrapper(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const T* alpha, const T* A, int lda, const T* x, int incx, const T* beta, T* y, int incy);

template <>
inline cublasStatus_t cublasGemv_wrapper<float>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) {
    return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline cublasStatus_t cublasGemv_wrapper<double>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) {
    return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <typename T>
cublasStatus_t cublasGemm_wrapper(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T* alpha, const T* A, int lda, const T* B, int ldb, const T* beta, T* C, int ldc);

template <>
inline cublasStatus_t cublasGemm_wrapper<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline cublasStatus_t cublasGemm_wrapper<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
cusolverStatus_t cusolverDnGetrf_bufferSize_wrapper(cusolverDnHandle_t handle, int m, int n, T* A, int lda, int* Lwork);

template <>
inline cusolverStatus_t cusolverDnGetrf_bufferSize_wrapper<float>(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork) {
    return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

template <>
inline cusolverStatus_t cusolverDnGetrf_bufferSize_wrapper<double>(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork) {
    return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

template <typename T>
cusolverStatus_t cusolverDnGetrf_wrapper(cusolverDnHandle_t handle, int m, int n, T* A, int lda, T* Workspace, int* DevIpiv, int* devInfo);

template <>
inline cusolverStatus_t cusolverDnGetrf_wrapper<float>(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* DevIpiv, int* devInfo) {
    return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, DevIpiv, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnGetrf_wrapper<double>(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* DevIpiv, int* devInfo) {
    return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, DevIpiv, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDnGetrs_wrapper(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const T* A, int lda, const int* devIpiv, T* B, int ldb, int* devInfo);

template <>
inline cusolverStatus_t cusolverDnGetrs_wrapper<float>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo) {
    return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnGetrs_wrapper<double>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo) {
    return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

// ==========================================
// 1. 结构体与类型定义
// ==========================================
template<typename T>
struct ZernikeOrderParam {
    int m;
    int n;
    int abs_m;
    int N;                       // 多项式最高次数 N = (n - abs_m) / 2
    T norm;                      // 归一化系数
    T coeffs[MAX_ZERNIKE_COEFFS]; // 从 c_0 到 c_N 的多项式系数
};

// ==========================================
// 2. 普通 Host 函数与 Kernel 的声明（只有声明，结尾是 ';'）
// ==========================================
__host__ double factorial(int k);

__host__ std::vector<int2> zernike_indices(int n_order);

__global__ void generate_zernike_basis_kernel(
    double* __restrict__ rho,
    double* __restrict__ theta,
    double* __restrict__ basis,
    int2* __restrict__ mn_indices,
    int H, int W, int n_order, bool is_norm);

// ==========================================
// 3. 模板函数 & 模板 Kernel（必须全部写在 .cuh 中）
// ==========================================

template<typename T>
__device__ __forceinline__ T pow_int(T base, int n) {
    T res = static_cast<T>(1.0);
    while (n > 0) {
        if (n & 1) res *= base;
        base *= base;
        n >>= 1;
    }
    return res;
}

// Host 端辅助模板函数：生成 Zernike 预计算参数
template<typename T>
inline std::vector<ZernikeOrderParam<T>> create_zernike_params(const std::vector<int2>& mn_indices) {
    std::vector<ZernikeOrderParam<T>> params(mn_indices.size());
    for (size_t o = 0; o < mn_indices.size(); ++o) {
        int m = mn_indices[o].x;
        int n = mn_indices[o].y;
        int abs_m = std::abs(m);

        ZernikeOrderParam<T> p = {};
        p.n = n;
        p.m = m;
        p.abs_m = abs_m;

        int N = (n - abs_m) / 2;
        p.N = N;

        for (int k = 0; k <= N; ++k) {
            int s = N - k; // 对应标准展开式的 (-1)^s 项
            double num = factorial(n - s);
            double den = factorial(s) * factorial((n + abs_m) / 2 - s) * factorial((n - abs_m) / 2 - s);
            double val = num / den;
            if (s % 2 != 0) val = -val;
            p.coeffs[k] = static_cast<T>(val);
        }

        double norm = (m == 0) ? std::sqrt(n + 1.0) : std::sqrt(2.0 * (n + 1.0));
        p.norm = static_cast<T>(norm);

        params[o] = p;
    }
    return params;
}

// 模板 Kernel 实现
template<typename T, bool MIX_PRECISION>
__global__ void generate_zernike_basis_kernel_optimized(
    const T* __restrict__ rho,
    const T* __restrict__ theta,
    T* __restrict__ basis,
    const ZernikeOrderParam<T>* __restrict__ params,
    int H, int W, int n_order, bool is_norm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = H * W;
    if (idx >= total_pixels) return;

    T r = rho[idx];

    // 超出单位圆直接置零
    if (r > static_cast<T>(1.0)) {
        for (int o = 0; o < n_order; ++o) {
            basis[o * total_pixels + idx] = static_cast<T>(0.0);
        }
        return;
    }

    T t = theta[idx];

    for (int o = 0; o < n_order; ++o) {
        const ZernikeOrderParam<T>& p = params[o];

        T radial;

        if constexpr (MIX_PRECISION) {
            double u = (double)r * (double)r;
            double poly = (double)p.coeffs[p.N];
            for (int k = p.N - 1; k >= 0; --k) {
                poly = fma(poly, u, (double)p.coeffs[k]);
            }
            radial = static_cast<T>(poly * pow_int((double)r, p.abs_m));
        }
        else {
            T u = r * r;
            T poly = p.coeffs[p.N];
            for (int k = p.N - 1; k >= 0; --k) {
                poly = fma(poly, u, p.coeffs[k]);
            }
            radial = poly * pow_int(r, p.abs_m);
        }

        T val = static_cast<T>(0.0);
        if (p.m > 0) {
            val = radial * cos(static_cast<T>(p.m) * t);
        }
        else if (p.m < 0) {
            val = radial * sin(static_cast<T>(p.abs_m) * t);
        }
        else {
            val = radial;
        }

        if (is_norm) {
            val *= p.norm;
        }

        basis[o * total_pixels + idx] = val;
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: 标记哪些像素满足 threshold 条件 (生成 uint8 Flag 标记)
// -----------------------------------------------------------------------------
template <typename T>
__global__ void generate_mask_flags_kernel(const T* d_rho,
    T threshold,
    uint8_t* d_flags,
    int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        d_flags[idx] = (d_rho[idx] <= threshold) ? 1 : 0;
    }
}

// -----------------------------------------------------------------------------
// CUDA Kernel: 提取 masked 内的 Zernike 基矩阵 B，尺寸为 (N, M)
// 此时 d_mask_indices 保证严格升序，访存能够高度合并！
// -----------------------------------------------------------------------------
template <typename T>
__global__ void gather_mask_kernel(const T* d_data,
    const int* d_mask_indices,
    T* d_B,
    int N,
    int M,
    int total_pixels) {
    int m = blockIdx.x * blockDim.x + threadIdx.x; // 0 ... M-1
    int n = blockIdx.y * blockDim.y + threadIdx.y; // 0 ... N-1

    if (m < M && n < N) {
        int pixel_idx = d_mask_indices[m];
        // 连续的 m 读取连续的 pixel_idx，高度命中 Cache 并触发连续 Block Read
        d_B[n * M + m] = d_data[n * total_pixels + pixel_idx];
    }
}

template <typename T>
void generate_mask(int*& d_mask_indices, const T* d_rho, T rho_threshold, int H, int W, int* h_M) {
    int total_pixels = H * W;
    uint8_t* d_flags = nullptr;
    int* d_M = nullptr;

    // 分配索引输出内存与数量计数器内存
    cudaMalloc(&d_flags, total_pixels * sizeof(uint8_t));
    cudaMalloc(&d_mask_indices, total_pixels * sizeof(int));
    cudaMalloc(&d_M, sizeof(int));

    // 2.1 生成标志位数组
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    generate_mask_flags_kernel<T> << <blocksPerGrid, threadsPerBlock >> > (
        d_rho, rho_threshold, d_flags, total_pixels
        );

    // 2.3 使用 thrust::counting_iterator 作为输入序列 (0, 1, 2, ..., total_pixels - 1)
    // 技巧：避免为了存储 0~total_pixels-1 的索引数组而额外分配显存
    thrust::counting_iterator<int> d_in_indices(0);

    // 2.4 查询 CUB 紧凑化 (Stream Compaction) 所需临时 Workspace 内存
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        d_in_indices, d_flags, d_mask_indices, d_M, total_pixels
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 2.5 执行 CUB 紧凑化提取（保留元素的原始相对顺序 -> 严格升序）
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        d_in_indices, d_flags, d_mask_indices, d_M, total_pixels);

    // 读取有效像素点数量 M
    cudaMemcpy(h_M, d_M, sizeof(int), cudaMemcpyDeviceToHost);

    // 释放标记图与 CUB 临时空间
    cudaFree(d_flags);
    cudaFree(d_temp_storage);
    cudaFree(d_M);

    if (*h_M == 0) {
        std::cerr << "Warning: No pixels satisfy the mask condition!" << std::endl;
        cudaFree(d_mask_indices);
        d_mask_indices = nullptr;
    }
}

// 1. Scatter (散布) Kernel：根据 mask 索引将 1D 提取点写回 2D/1D 全图位置
template <typename T>
__global__ void scatter_mask_kernel(
    const T* __restrict__ src,       // 输入: d_img_sel [长度 M]
    const int* __restrict__ indices, // 输入: d_mask_indices [长度 M]
    T* __restrict__ dst,             // 输出: d_img_restored [长度 H * W]
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        dst[indices[idx]] = src[idx];
    }
}

// -----------------------------------------------------------------------------
// 函数: zerk_fit_pinv_generate_cuda
// -----------------------------------------------------------------------------
template <typename T>
void zerk_fit_pinv_generate_cuda(const T* d_zernike_basis,
    const T* d_rho,
    int* d_mask_indices,
    int N, int H, int W,
    T** d_pinv,
    int* h_M) {
    int total_pixels = H * W;

    // 1. 句柄初始化
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    cublasCreate(&cublasH);
    cusolverDnCreate(&cusolverH);

    // -------------------------------------------------------------------------
    // 3. 构建选择后的矩阵 B (N, M)，行优先
    // 此时 d_mask_indices 是升序的，读取 d_zernike_basis 时能够大幅提升缓存命中率
    // -------------------------------------------------------------------------
    int M = h_M[0];
    T* d_B = nullptr;
    cudaMalloc(&d_B, N * M * sizeof(T));

    dim3 gridDim_gather((M + 15) / 16, (N + 15) / 16);
    dim3 blockDim_gather(16, 16);
    gather_mask_kernel<T> << <gridDim_gather, blockDim_gather >> > (
        d_zernike_basis, d_mask_indices, d_B, N, M, total_pixels
        );

    // -------------------------------------------------------------------------
    // 4. 使用 cuBLAS 计算 G = B * B^T ，尺寸为 (N, N)
    // -------------------------------------------------------------------------
    T* d_G = nullptr;
    cudaMalloc(&d_G, N * N * sizeof(T));

    const T alpha = 1.0;
    const T beta = 0.0;

    cublasGemm_wrapper<T>(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, M,
        &alpha,
        d_B, M,        // B_CM (M, N), lda = M
        d_B, M,        // B_CM (M, N), ldb = M
        &beta,
        d_G, N         // G_CM (N, N), ldc = N
    );

    // -------------------------------------------------------------------------
    // 5. 使用 cuSOLVER 对 G (N, N) 求逆：
    //    1) 构造单位矩阵 I_N
    //    2) 用 Sgetrf 进行 LU 分解
    //    3) 用 Sgetrs 求解 G * X = I_N，解出的 X 即为 G^-1
    // -------------------------------------------------------------------------

    // 5.1 在 Host 端创建 N x N 单位矩阵，并拷贝到 GPU 端 d_G_inv
    T* d_G_inv = nullptr;
    cudaMalloc(&d_G_inv, N * N * sizeof(T));

    std::vector<T> h_identity(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        h_identity[i * N + i] = 1.0; // 对角线置 1.0
    }
    cudaMemcpy(d_G_inv, h_identity.data(), N * N * sizeof(T), cudaMemcpyHostToDevice);

    // 5.2 分配 LU 分解所需内存
    int lwork = 0;
    int* d_ipiv = nullptr, * d_info = nullptr;
    cudaMalloc(&d_ipiv, N * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));

    // 5.3 执行 LU 分解 (cusolverDnSgetrf)
    cusolverDnGetrf_bufferSize_wrapper<T>(cusolverH, N, N, d_G, N, &lwork);
    T* d_work = nullptr;
    cudaMalloc(&d_work, lwork * sizeof(T));

    cusolverDnGetrf_wrapper<T>(cusolverH, N, N, d_G, N, d_work, d_ipiv, d_info);

    // 5.4 使用 cusolverDnSgetrs 求解 G * X = I_N => X 为 G^-1 (结果直接覆写覆盖 d_G_inv)
    cusolverDnGetrs_wrapper<T>(
        cusolverH,
        CUBLAS_OP_N,
        N, N,           // 方程组阶数 N，右端向量/列数 N
        d_G, N,         // LU 分解后的矩阵 G
        d_ipiv,         // 主元索引
        d_G_inv, N,     // 输入为单位矩阵 I_N，计算完成后自动变为逆矩阵 G^-1
        d_info
    );

    // -------------------------------------------------------------------------
    // 6. 计算最终结果 P = G^-1 * B，尺寸为 (N, M)，行优先
    // -------------------------------------------------------------------------
    cudaMalloc(d_pinv, N * M * sizeof(T));

    cublasGemm_wrapper<T>(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, N,
        &alpha,
        d_B, M,        // B_CM (M, N), lda = M
        d_G_inv, N,    // G^-1_CM (N, N), ldb = N
        &beta,
        *d_pinv, M     // P_CM (M, N), ldc = M
    );

    // 7. 释放中间资源（记得加上 d_G_inv 的释放）
    cudaFree(d_B);
    cudaFree(d_G);
    cudaFree(d_G_inv); // 释放逆矩阵显存
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
}

// 波前重构与 NaN 清理融合 Kernel (einsum('i, ijk -> jk') + nan_to_num)
template<typename T>
__global__ void wavefront_recover_kernel(
    const T* __restrict__ zerk,
    const T* __restrict__ zernike_basis, // 形状: [num_modes, H, W]
    T* __restrict__ img_re,
    int start_mode,
    int end_mode,
    int H,
    int W
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        int spatial_idx = y * W + x;
        int spatial_size = H * W;
        T sum = 0.0;

        for (int i = start_mode; i < end_mode; ++i) {
            T coeff = zerk[i];
            T basis_val = zernike_basis[i * spatial_size + spatial_idx];
            sum += coeff * basis_val;
        }

        // 对应 np.nan_to_num 功能
        img_re[spatial_idx] = isnan(sum) ? 0.0 : sum;
    }
}

template<typename T>
void zernike_low_pass_filter(T* __restrict__ img, T* img_re, T* __restrict__ zernike_basis_pinv, T* __restrict__ zernike_basis,
    T* __restrict__ rho, T rho_threshold, int* d_mask_indices, int H, int W, int n_order, int* h_M, int start_mode, int end_mode) 
{
    int M = h_M[0];
    if (M <= 0) return;

    int total_pixels = H * W;

    // 1. 句柄初始化
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);

    // -------------------------------------------------------------------------
    // 3. 构建选择后的矩阵 B (N, M)，行优先
    // 此时 d_mask_indices 是升序的，读取 d_zernike_basis 时能够大幅提升缓存命中率
    // -------------------------------------------------------------------------
    T* d_img_sel = nullptr;
    cudaMalloc(&d_img_sel, M * sizeof(T));

    dim3 gridDim_gather((M + 15) / 16, (1 + 15) / 16);
    dim3 blockDim_gather(16, 16);
    gather_mask_kernel<T> << <gridDim_gather, blockDim_gather >> > (
        img, d_mask_indices, d_img_sel, 1, M, total_pixels
        );


    T* d_zerk = nullptr;
    cudaMalloc(&d_zerk, n_order * sizeof(T));

    const T alpha = 1.0;
    const T beta = 0.0;

    cublasGemv_wrapper<T>(
        cublasH,
        CUBLAS_OP_T,
        M, n_order,
        &alpha,
        zernike_basis_pinv, M,
        d_img_sel, 1,
        &beta,
        d_zerk, 1
    );

    // 3. 波前重构与清理
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);
    wavefront_recover_kernel<T> << <gridDim, blockDim >> > (d_zerk, zernike_basis, // 形状: [num_modes, H, W]
        img_re,
        start_mode,
        end_mode,
        H, W
    );


    cudaFree(d_img_sel);
    cudaFree(d_zerk);
    cublasDestroy(cublasH);

}

// 模板 Host 包装函数实现
template<typename T, bool MIX_PRECISION>
void generate_zernike_basis_optimized(
    const T* __restrict__ rho,
    const T* __restrict__ theta,
    T* __restrict__ basis,
    int H, int W, int n_order, bool is_norm)
{
    std::vector<int2> mn_indices = zernike_indices(n_order);
    ZernikeOrderParam<T>* d_params = nullptr;
    std::vector<ZernikeOrderParam<T>> h_params = create_zernike_params<T>(mn_indices);

    int2* d_mn_indices;

    cudaMalloc((void**)&d_mn_indices, n_order * sizeof(int2));
    cudaMalloc(&d_params, n_order * sizeof(ZernikeOrderParam<T>));

    cudaMemcpy(d_mn_indices, mn_indices.data(), n_order * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, h_params.data(), n_order * sizeof(ZernikeOrderParam<T>), cudaMemcpyHostToDevice);

    int threads = 512;
    int blocks = (H * W + threads - 1) / threads;
    generate_zernike_basis_kernel_optimized<T, MIX_PRECISION> << <blocks, threads >> > (
        rho, theta, basis, d_params, H, W, n_order, is_norm);

    cudaDeviceSynchronize();

    cudaFree(d_mn_indices);
    cudaFree(d_params);
}