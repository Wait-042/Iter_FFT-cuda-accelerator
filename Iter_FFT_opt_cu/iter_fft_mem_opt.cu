#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cusolver.lib")
#pragma comment(lib, "cufft.lib")

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define M_PI 3.14159265358979323846
#define NUM_STREAMS 3
#define MAX_ZERNIKE_COEFFS 10
#define MAX_PRECOMPUTE_M 16 // 用于优化 Zernike 角度和半径幂运算的寄存器缓存大小，支持极高阶

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

inline size_t align256(size_t bytes) {
    return (bytes + 255) & ~((size_t)255);
}

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 1)
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define PREFETCH_READ(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define PREFETCH_READ(addr)
#endif

template <typename T>
struct PhaseConstants;

template <>
struct PhaseConstants<float> {
    static constexpr float PI = 3.14159265358979323846f;
    static constexpr float TWO_PI = 6.28318530717958647692f;
};

template <>
struct PhaseConstants<double> {
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double TWO_PI = 6.28318530717958647692;
};

// ==========================================
// cufft 模板封装
// ==========================================
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

// ==========================================
// cuBLAS / cuSOLVER 模板 Wrapper 封装
// ==========================================
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
inline cusolverStatus_t cusolverDnPotrf_bufferSize_wrapper(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T* A, int lda, int* lwork);

template <>
inline cusolverStatus_t cusolverDnPotrf_bufferSize_wrapper<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork) {
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
}

template <>
inline cusolverStatus_t cusolverDnPotrf_bufferSize_wrapper<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork) {
    return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
}

template <typename T>
inline cusolverStatus_t cusolverDnPotrf_wrapper(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T* A, int lda, T* Workspace, int Lwork, int* devInfo);

template <>
inline cusolverStatus_t cusolverDnPotrf_wrapper<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* Workspace, int Lwork, int* devInfo) {
    return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnPotrf_wrapper<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* Workspace, int Lwork, int* devInfo) {
    return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

template <typename T>
inline cusolverStatus_t cusolverDnPotrs_wrapper(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const T* A, int lda, T* B, int ldb, int* devInfo);

template <>
inline cusolverStatus_t cusolverDnPotrs_wrapper<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* devInfo) {
    return cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnPotrs_wrapper<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* devInfo) {
    return cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

__device__ __forceinline__ float my_sqrt(float v) { return sqrtf(v); }
__device__ __forceinline__ double my_sqrt(double v) { return sqrt(v); }

__device__ __forceinline__ float my_atan2(float y, float x) { return atan2f(y, x); }
__device__ __forceinline__ double my_atan2(double y, double x) { return atan2(y, x); }

// 频谱截取窗口结构体
struct Rect {
    int y_min, y_max;
    int x_min, x_max;
};

// zernike多项式参数结构体
template<typename T>
struct ZernikeOrderParam {
    int m;
    int n;
    int abs_m;
    int N;
    T norm;
    T coeffs[MAX_ZERNIKE_COEFFS];
};


// 相位解包裹边结构体
struct CompactEdge { int u, v; int k0; };

struct DSUNode {
    int parent;   // 父节点索引
    int offset;   // 到父节点的相对偏移量
    int rank;     // 树的高度（按秩合并用）
};

struct FastDSU {
    DSUNode* nodes; // 节点数组指针

    // 初始化：每个节点自成集合，父节点为自身
    inline void init(int n) {
        for (int i = 0; i < n; ++i) {
            nodes[i] = { i, 0, 0 };
        }
    }

    // 带路径压缩和偏移累加的查找操作
    inline int find(int i, int& total_offset) {
        DSUNode* __restrict__ pnodes = nodes;
        int curr = i;
        int acc_off = 0;

        // 第一遍：沿父链向上查找根节点，同时累加路径上的偏移量
        while (curr != pnodes[curr].parent) {
            acc_off += pnodes[curr].offset;
            curr = pnodes[curr].parent;
        }
        int root = curr;
        total_offset = acc_off; // 返回节点i到根的总偏移

        // 第二遍：路径压缩，将路径上所有节点直接挂到根节点下，并更新偏移
        curr = i;
        int path_off = 0;
        while (curr != root) {
            int next = pnodes[curr].parent;             // 暂存原父节点
            int old_off = pnodes[curr].offset;          // 暂存原偏移
            pnodes[curr].parent = root;                 // 直接指向根
            pnodes[curr].offset = acc_off - path_off;   // 更新为到根的正确偏移
            path_off += old_off;                        // 累计已走过的偏移
            curr = next;
        }
        return root;
    }
};

template <typename T_data, typename T_phase>
struct PipelineWorkspace {
    int H, W, N, N_ORDER;
    int sel_pixels;

    char* d_gpu_pool = nullptr;
    char* h_host_pool = nullptr;

    cudaStream_t streams[NUM_STREAMS];
    cublasHandle_t cublas_handles[NUM_STREAMS];
    cusolverDnHandle_t cusolver_handle;
    cufftHandle fft_plans[NUM_STREAMS];
    cudaEvent_t fft_done_event;

    typename CuFFTTraits<T_data>::Complex* d_img_complex;
    typename CuFFTTraits<T_data>::Complex* d_fft_a;
    typename CuFFTTraits<T_data>::Complex* d_fft_bx;
    typename CuFFTTraits<T_data>::Complex* d_fft_by;

    T_data* d_rho;         T_data* d_theta;  ZernikeOrderParam<T_data>* d_zernike_params;
    int* d_mask_indices;   uint8_t* d_flags;

    T_data* d_a_fft;       T_data* d_bx_fft;       T_data* d_by_fft;
    T_data* d_a_model;     T_data* d_bx_model;     T_data* d_by_model;
    T_data* d_a_fft_model; T_data* d_bx_fft_model; T_data* d_by_fft_model;
    T_data* d_a_corr;      T_data* d_bx_corr;      T_data* d_by_corr;

    T_phase* d_phix_fft;   T_phase* d_phiy_fft;
    T_phase* d_phix_model; T_phase* d_phiy_model;
    T_phase* d_phix_fft_model; T_phase* d_phiy_fft_model;
    T_phase* d_phix; T_phase* d_phiy;
    T_phase* d_img; T_phase* d_zernike_basis;

    T_data* d_img_sel_5ch = nullptr;
    T_data* d_zerk_5ch = nullptr;
    T_data* d_img_re_5ch = nullptr;

    // --- CUB RadixSort DoubleBuffer 零动态分配内存支持 ---
    CompactEdge* d_edges;
    CompactEdge* d_edges_alt;
    uint32_t* d_keys;        // 32-bit 优化压缩 Key
    uint32_t* d_keys_alt;
    T_phase* d_unwrap_reliability;
    int* d_unwrap_offset;
    DSUNode* d_dsu_nodes;

    T_data* d_pinv;
    T_data* d_fit_B;
    T_data* d_fit_G;
    T_data* d_fit_G_inv;
    T_data* d_cusolver_work;
    int* d_cusolver_info;

    void* d_cub_temp_storage;
    size_t cub_temp_storage_bytes = 32 * 1024 * 1024; // CUB 临时空间
    int* d_num_selected_out;

    CompactEdge* h_pinned_edges;
    DSUNode* h_dsu_nodes;

    void init(int height, int width, int max_n_order) {
        H = height; W = width; N = H * W; N_ORDER = max_n_order;

        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cublasCreate(&cublas_handles[i]);
            cublasSetStream(cublas_handles[i], streams[i]);
            cufftPlan2d(&fft_plans[i], H, W, CuFFTTraits<T_data>::C2C_TYPE);
            cufftSetStream(fft_plans[i], streams[i]);
        }
        cusolverDnCreate(&cusolver_handle);
        cusolverDnSetStream(cusolver_handle, streams[0]);
        cudaEventCreateWithFlags(&fft_done_event, cudaEventDisableTiming);

        size_t data_bytes = N * sizeof(T_data);
        size_t phase_bytes = N * sizeof(T_phase);
        size_t complex_bytes = N * sizeof(typename CuFFTTraits<T_data>::Complex);
        int max_edges = H * (W - 1) + (H - 1) * W;

        size_t total_gpu_bytes = 0;
        auto reserve_gpu = [&](size_t size) -> size_t {
            size_t offset = total_gpu_bytes;
            total_gpu_bytes += align256(size);
            return offset;
            };

        size_t off_img_complex = reserve_gpu(complex_bytes);
        size_t off_fft_a = reserve_gpu(complex_bytes);
        size_t off_fft_bx = reserve_gpu(complex_bytes);
        size_t off_fft_by = reserve_gpu(complex_bytes);

        size_t off_rho = reserve_gpu(data_bytes);
        size_t off_theta = reserve_gpu(data_bytes);
        size_t off_zernike_params = reserve_gpu(N_ORDER * sizeof(ZernikeOrderParam<T_data>));
        size_t off_mask_indices = reserve_gpu(N * sizeof(int));
        size_t off_flags = reserve_gpu(N * sizeof(uint8_t));

        size_t off_a_fft = reserve_gpu(data_bytes);
        size_t off_bx_fft = reserve_gpu(data_bytes);
        size_t off_by_fft = reserve_gpu(data_bytes);
        size_t off_a_model = reserve_gpu(data_bytes);
        size_t off_bx_model = reserve_gpu(data_bytes);
        size_t off_by_model = reserve_gpu(data_bytes);
        size_t off_a_fft_model = reserve_gpu(data_bytes);
        size_t off_bx_fft_model = reserve_gpu(data_bytes);
        size_t off_by_fft_model = reserve_gpu(data_bytes);
        size_t off_a_corr = reserve_gpu(data_bytes);
        size_t off_bx_corr = reserve_gpu(data_bytes);
        size_t off_by_corr = reserve_gpu(data_bytes);

        size_t off_phix = reserve_gpu(phase_bytes);
        size_t off_phiy = reserve_gpu(phase_bytes);
        size_t off_img = reserve_gpu(phase_bytes);
        size_t off_zernike_basis = reserve_gpu(phase_bytes * max_n_order);
        size_t off_phix_fft = reserve_gpu(phase_bytes);
        size_t off_phiy_fft = reserve_gpu(phase_bytes);
        size_t off_phix_model = reserve_gpu(phase_bytes);
        size_t off_phiy_model = reserve_gpu(phase_bytes);
        size_t off_phix_fft_model = reserve_gpu(phase_bytes);
        size_t off_phiy_fft_model = reserve_gpu(phase_bytes);

        size_t off_img_sel_5ch = reserve_gpu(5 * N * sizeof(T_data));
        size_t off_zerk_5ch = reserve_gpu(5 * max_n_order * sizeof(T_data));
        size_t off_img_re_5ch = reserve_gpu(5 * N * sizeof(T_data));

        size_t off_edges = reserve_gpu(max_edges * sizeof(CompactEdge));
        size_t off_edges_alt = reserve_gpu(max_edges * sizeof(CompactEdge));
        size_t off_keys = reserve_gpu(max_edges * sizeof(uint32_t));
        size_t off_keys_alt = reserve_gpu(max_edges * sizeof(uint32_t));
        size_t off_unwrap_reliability = reserve_gpu(N * sizeof(T_phase));
        size_t off_unwrap_offset = reserve_gpu(N * sizeof(int));
        size_t off_dsu_nodes = reserve_gpu(N * sizeof(DSUNode));

        size_t off_pinv = reserve_gpu(N_ORDER * N * sizeof(T_data));
        size_t off_fit_B = reserve_gpu(N_ORDER * N * sizeof(T_data));
        size_t off_fit_G = reserve_gpu(N_ORDER * N_ORDER * sizeof(T_data));
        size_t off_fit_G_inv = reserve_gpu(N_ORDER * N_ORDER * sizeof(T_data));
        size_t off_cusolver_work = reserve_gpu(N_ORDER * N_ORDER * sizeof(T_data));
        size_t off_cusolver_info = reserve_gpu(sizeof(int));

        size_t off_cub_temp_storage = reserve_gpu(cub_temp_storage_bytes);
        size_t off_num_selected_out = reserve_gpu(sizeof(int));

        CUDA_CHECK(cudaMalloc(&d_gpu_pool, total_gpu_bytes));

        size_t total_host_bytes = align256(max_edges * sizeof(CompactEdge)) +
            align256(N * sizeof(int)) +
            align256(N * sizeof(DSUNode));
        CUDA_CHECK(cudaHostAlloc(&h_host_pool, total_host_bytes, cudaHostAllocDefault));
        size_t h_off1 = 0;
        size_t h_off2 = h_off1 + align256(max_edges * sizeof(CompactEdge));
        size_t h_off3 = h_off2 + align256(N * sizeof(int));

        h_pinned_edges = reinterpret_cast<CompactEdge*>(h_host_pool + h_off1);
        h_dsu_nodes = reinterpret_cast<DSUNode*>(h_host_pool + h_off3);

        d_img_complex = reinterpret_cast<typename CuFFTTraits<T_data>::Complex*>(d_gpu_pool + off_img_complex);
        d_fft_a = reinterpret_cast<typename CuFFTTraits<T_data>::Complex*>(d_gpu_pool + off_fft_a);
        d_fft_bx = reinterpret_cast<typename CuFFTTraits<T_data>::Complex*>(d_gpu_pool + off_fft_bx);
        d_fft_by = reinterpret_cast<typename CuFFTTraits<T_data>::Complex*>(d_gpu_pool + off_fft_by);

        d_rho = reinterpret_cast<T_data*>(d_gpu_pool + off_rho);
        d_theta = reinterpret_cast<T_data*>(d_gpu_pool + off_theta);
        d_zernike_params = reinterpret_cast<ZernikeOrderParam<T_data>*>(d_gpu_pool + off_zernike_params);
        d_mask_indices = reinterpret_cast<int*>(d_gpu_pool + off_mask_indices);
        d_flags = reinterpret_cast<uint8_t*>(d_gpu_pool + off_flags);

        d_a_fft = reinterpret_cast<T_data*>(d_gpu_pool + off_a_fft);
        d_bx_fft = reinterpret_cast<T_data*>(d_gpu_pool + off_bx_fft);
        d_by_fft = reinterpret_cast<T_data*>(d_gpu_pool + off_by_fft);
        d_a_model = reinterpret_cast<T_data*>(d_gpu_pool + off_a_model);
        d_bx_model = reinterpret_cast<T_data*>(d_gpu_pool + off_bx_model);
        d_by_model = reinterpret_cast<T_data*>(d_gpu_pool + off_by_model);
        d_a_fft_model = reinterpret_cast<T_data*>(d_gpu_pool + off_a_fft_model);
        d_bx_fft_model = reinterpret_cast<T_data*>(d_gpu_pool + off_bx_fft_model);
        d_by_fft_model = reinterpret_cast<T_data*>(d_gpu_pool + off_by_fft_model);
        d_a_corr = reinterpret_cast<T_data*>(d_gpu_pool + off_a_corr);
        d_bx_corr = reinterpret_cast<T_data*>(d_gpu_pool + off_bx_corr);
        d_by_corr = reinterpret_cast<T_data*>(d_gpu_pool + off_by_corr);

        d_phix = reinterpret_cast<T_phase*>(d_gpu_pool + off_phix);
        d_phiy = reinterpret_cast<T_phase*>(d_gpu_pool + off_phiy);
        d_img = reinterpret_cast<T_phase*>(d_gpu_pool + off_img);
        d_zernike_basis = reinterpret_cast<T_phase*>(d_gpu_pool + off_zernike_basis);
        d_phix_fft = reinterpret_cast<T_phase*>(d_gpu_pool + off_phix_fft);
        d_phiy_fft = reinterpret_cast<T_phase*>(d_gpu_pool + off_phiy_fft);
        d_phix_model = reinterpret_cast<T_phase*>(d_gpu_pool + off_phix_model);
        d_phiy_model = reinterpret_cast<T_phase*>(d_gpu_pool + off_phiy_model);
        d_phix_fft_model = reinterpret_cast<T_phase*>(d_gpu_pool + off_phix_fft_model);
        d_phiy_fft_model = reinterpret_cast<T_phase*>(d_gpu_pool + off_phiy_fft_model);

        d_img_sel_5ch = reinterpret_cast<T_data*>(d_gpu_pool + off_img_sel_5ch);
        d_zerk_5ch = reinterpret_cast<T_data*>(d_gpu_pool + off_zerk_5ch);
        d_img_re_5ch = reinterpret_cast<T_data*>(d_gpu_pool + off_img_re_5ch);

        d_edges = reinterpret_cast<CompactEdge*>(d_gpu_pool + off_edges);
        d_edges_alt = reinterpret_cast<CompactEdge*>(d_gpu_pool + off_edges_alt);
        d_keys = reinterpret_cast<uint32_t*>(d_gpu_pool + off_keys);
        d_keys_alt = reinterpret_cast<uint32_t*>(d_gpu_pool + off_keys_alt);
        d_unwrap_reliability = reinterpret_cast<T_phase*>(d_gpu_pool + off_unwrap_reliability);
        d_unwrap_offset = reinterpret_cast<int*>(d_gpu_pool + off_unwrap_offset);
        d_dsu_nodes = reinterpret_cast<DSUNode*>(d_gpu_pool + off_dsu_nodes);

        d_pinv = reinterpret_cast<T_data*>(d_gpu_pool + off_pinv);
        d_fit_B = reinterpret_cast<T_data*>(d_gpu_pool + off_fit_B);
        d_fit_G = reinterpret_cast<T_data*>(d_gpu_pool + off_fit_G);
        d_fit_G_inv = reinterpret_cast<T_data*>(d_gpu_pool + off_fit_G_inv);
        d_cusolver_work = reinterpret_cast<T_data*>(d_gpu_pool + off_cusolver_work);
        d_cusolver_info = reinterpret_cast<int*>(d_gpu_pool + off_cusolver_info);

        d_cub_temp_storage = reinterpret_cast<void*>(d_gpu_pool + off_cub_temp_storage);
        d_num_selected_out = reinterpret_cast<int*>(d_gpu_pool + off_num_selected_out);
    }

    void destroy() {
        cudaEventDestroy(fft_done_event);
        cusolverDnDestroy(cusolver_handle);
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cufftDestroy(fft_plans[i]);
            cublasDestroy(cublas_handles[i]);
            cudaStreamDestroy(streams[i]);
        }
        if (d_gpu_pool) cudaFree(d_gpu_pool);
        if (h_host_pool) cudaFreeHost(h_host_pool);
    }
};

template <typename T>
void save_raw(const std::string& filename, const std::vector<T>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        size_t total_bytes = data.size() * sizeof(T);
        file.write(reinterpret_cast<const char*>(data.data()), total_bytes);
        file.close();
        std::cout << "save successful " << filename << " (file size: " << total_bytes / (1024.0 * 1024.0) << " MB)" << std::endl;
    }
    else {
        std::cerr << "err: can't open file " << filename << " to write!" << std::endl;
    }
}


// 系统误差生成 Kernel：计算归一化坐标、极坐标、光程差相位及模拟图像
template <typename T, typename T_phase>
__global__ void system_error_generate(
    T norm_val,       // 归一化起始偏移值
    T step,           // 归一化坐标步进
    T cmos_radius,    // CMOS 传感器物理半径
    T ddx, T ddy, T ddz, // 光源/参考点在 x, y, z 方向的偏移量
    T scale_factor,   // 光程差到相位的缩放因子
    int M,            // 图像/网格尺寸 (M x M)
    PipelineWorkspace<T, T_phase> ws) // 包含各类输出缓冲区的管道工作空间 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * M + col;

    if (row >= M || col >= M) return;

    // 计算归一化坐标及极坐标 (rho: 半径, theta: 角度)
    T x_norm = static_cast<T>(-norm_val + col * step);
    T y_norm = static_cast<T>(-norm_val + row * step);
    T rho = static_cast<T>(my_sqrt(x_norm * x_norm + y_norm * y_norm));
    ws.d_rho[idx] = rho;
    ws.d_theta[idx] = static_cast<T>(my_atan2(y_norm, x_norm));

    // 将网格坐标映射为 CMOS 传感器上的实际物理坐标 (xx, yy)
    T xx = static_cast<T>((-1.0 + 2.0 * col / (M - 1)) * cmos_radius);
    T yy = static_cast<T>((-1.0 + 2.0 * row / (M - 1)) * cmos_radius);
    T xx_r2 = xx * xx;
    T yy_r2 = yy * yy;
    T ddz_r2 = ddz * ddz;

    // 计算当前点到 x/y 方向偏移光源的距离，以及到参考原点的距离
    T dist_x = sqrt((xx - ddx) * (xx - ddx) + yy_r2 + ddz_r2);
    T dist_y = sqrt(xx_r2 + (yy - ddy) * (yy - ddy) + ddz_r2);
    T dist_ref = sqrt(xx_r2 + yy_r2 + ddz_r2);

    // 利用近似公式计算 x/y 方向的光程差 (OPD)
    T diff_x = (ddx * ddx - static_cast<T>(2.0) * xx * ddx) / (dist_x + dist_ref);
    T diff_y = (ddy * ddy - static_cast<T>(2.0) * yy * ddy) / (dist_y + dist_ref);

    // 将光程差乘以缩放因子转换为相位值
    T_phase x_val = static_cast<T_phase>(diff_x * scale_factor);
    T_phase y_val = static_cast<T_phase>(diff_y * scale_factor);

    // 写入相位缓冲，并根据极坐标半径生成掩膜图像
    // 仅在有效圆形区域 (rho <= 1.0) 内生成干涉图像，外部置零
    ws.d_phix[idx] = x_val;
    ws.d_phiy[idx] = y_val;
    if (rho <= static_cast<T>(1.0)) {
        ws.d_img[idx] = static_cast<T_phase>(1.0) + cos(x_val) + cos(y_val);
    }
    else {
        ws.d_img[idx] = static_cast<T_phase>(0.0);
    }
}

// 整数快速幂：通过位运算实现 O(log n) 复杂度的幂运算，避免调用缓慢的通用 pow 函数
template<typename T>
__device__ __forceinline__ T pow_int(T base, int n) {
    T res = static_cast<T>(1.0);
    while (n > 0) {
        if (n & 1) res *= base; // 若当前位为1，则累乘到结果中
        base *= base;
        n >>= 1;
    }
    return res;
}

// 阶乘计算：返回 k! 的值（注意：k 过大时易导致 double 溢出或精度丢失, 当前项目zernike阶数通常不会太大）
__device__ __forceinline__ double factorial(int k) {
    double res = 1.0;
    for (int i = 2; i <= k; ++i) res *= i;
    return res;
}

// 预计算 Zernike 多项式参数 Kernel：为每个 Zernike 阶数提前计算径向多项式系数及归一化因子
template<typename T, typename T_phase>
__global__ void create_zernike_params(PipelineWorkspace<T, T_phase> ws) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ws.N_ORDER) return;

    // 根据 OSA (Optical Society of America) 标准单索引，反解出径向阶数 n 和角向频率 m
    int order_idx = idx + 1;
    int d = static_cast<int>(floor(sqrt(static_cast<double>(order_idx - 1)))) + 1;
    int m = (((d * d - order_idx) & 1) != 0) ? (-d * d + order_idx - 1) / 2 : (d * d - order_idx) / 2;
    int n = 2 * (d - 1) - abs(m);
    int abs_m = abs(m);

    ZernikeOrderParam<T> p = {};
    p.n = n;
    p.m = m;
    p.abs_m = abs_m;

    // 计算径向多项式 R_n^m(rho) 的展开项系数
    // 公式基于阶乘组合：coeffs[k] = (-1)^s * (n-s)! / [s! * ((n+|m|)/2-s)! * ((n-|m|)/2-s)!]
    int N = (n - abs_m) / 2;
    p.N = N;

    for (int k = 0; k <= N; ++k) {
        int s = N - k;
        double num = factorial(n - s);
        double den = factorial(s) * factorial((n + abs_m) / 2 - s) * factorial((n - abs_m) / 2 - s);
        double val = num / den;
        if (s % 2 != 0) val = -val;
        p.coeffs[k] = static_cast<T>(val);
    }

    // 计算 Zernike 多项式的正交归一化因子
    // m=0 时归一化系数为 sqrt(n+1)，m!=0 时为 sqrt(2*(n+1))
    double norm = (m == 0) ? sqrt(static_cast<double>(n + 1)) : sqrt(2.0 * static_cast<double>(n + 1));
    p.norm = static_cast<T>(norm);

    ws.d_zernike_params[idx] = p;
}

// -------------------------------------------------------------
// [优化版] Zernike 基底生成内核
// 引入优化策略:
// 1. Shared Memory 动态装载 Zernike 参数
// 2. 切比雪夫局部寄存器缓存，用递推消除所有 64 阶硬件三角函数 `cos/sin` 调用
// 3. 半径幂局部缓存消除 `pow_int` 冗余运算
// -------------------------------------------------------------
template<typename T, typename T_phase>
__global__ void generate_zernike_basis_kernel(PipelineWorkspace<T, T_phase> ws, bool is_norm) {
    // 动态分配 Shared Memory
    extern __shared__ char s_mem[];
    ZernikeOrderParam<T>* s_params = reinterpret_cast<ZernikeOrderParam<T>*>(s_mem);

    int n_order = ws.N_ORDER;

    // 1. Block 内所有线程协作并行读取全局内存到共享内存
    for (int i = threadIdx.x; i < n_order; i += blockDim.x) {
        s_params[i] = ws.d_zernike_params[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = ws.N;

    if (idx >= total_pixels) return;

    T r = ws.d_rho[idx];

    if (r > static_cast<T>(1.0)) {
        for (int o = 0; o < n_order; ++o) {
            ws.d_zernike_basis[o * total_pixels + idx] = static_cast<T>(0.0);
        }
        return;
    }

    T t = ws.d_theta[idx];
    T u = r * r;

    // 2. 利用切比雪夫递推预计算三角函数与极径幂 (大幅降低 ALU 开销)
    T cos_mt[MAX_PRECOMPUTE_M];
    T sin_mt[MAX_PRECOMPUTE_M];
    T r_m[MAX_PRECOMPUTE_M];

    cos_mt[0] = static_cast<T>(1.0);
    sin_mt[0] = static_cast<T>(0.0);
    r_m[0] = static_cast<T>(1.0);

    cos_mt[1] = cos(t);
    sin_mt[1] = sin(t);
    r_m[1] = r;

#pragma unroll
    for (int m = 2; m < MAX_PRECOMPUTE_M; ++m) {
        cos_mt[m] = cos_mt[m - 1] * cos_mt[1] - sin_mt[m - 1] * sin_mt[1];
        sin_mt[m] = sin_mt[m - 1] * cos_mt[1] + cos_mt[m - 1] * sin_mt[1];
        r_m[m] = r_m[m - 1] * r;
    }

    // 3. 极速计算内核
    for (int o = 0; o < n_order; ++o) {
        // 从 Shared Memory 极速读取 Zernike 参数结构体，自带多播 broadcast，不会冲突
        const ZernikeOrderParam<T>& p = s_params[o];
        T poly = p.coeffs[p.N];

#pragma unroll 4
        for (int k = p.N - 1; k >= 0; --k) {
            poly = fma(poly, u, p.coeffs[k]);
        }

        T val = static_cast<T>(0.0);
        int am = p.abs_m;

        // 走高速 Cache 路径
        if (am < MAX_PRECOMPUTE_M) {
            T radial = poly * r_m[am];
            if (p.m > 0) {
                val = radial * cos_mt[am];
            }
            else if (p.m < 0) {
                val = radial * sin_mt[am];
            }
            else {
                val = radial;
            }
        }
        else {
            // 超高阶的 Fallback 回退逻辑（应对 N_ORDER 极大场景）
            T radial = poly * pow_int(r, am);
            if (p.m > 0) {
                val = radial * cos(static_cast<T>(am) * t);
            }
            else if (p.m < 0) {
                val = radial * sin(static_cast<T>(am) * t);
            }
            else {
                val = radial;
            }
        }

        if (is_norm) {
            val *= p.norm;
        }

        ws.d_zernike_basis[o * total_pixels + idx] = val;
    }
}

// 掩膜标志生成 Kernel：根据极坐标半径阈值生成二值掩膜 (1:有效, 0:无效)
template <typename T>
__global__ void generate_mask_flags_kernel(const T* d_rho, T threshold, uint8_t* d_flags, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        d_flags[idx] = (d_rho[idx] <= threshold) ? 1 : 0;
    }
}

// 掩膜生成与有效像素索引提取：结合 CUB 库筛选出满足掩膜条件的像素一维索引
template <typename T, typename T_phase>
void generate_mask(PipelineWorkspace<T, T_phase>& ws, T rho_threshold, int& sel_pixels) {
    int total_pixels = ws.N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    nvtxRangePushA("generate_mask_flags_kernel");
    // 1. 生成二值掩膜标志
    generate_mask_flags_kernel<T> << <blocksPerGrid, threadsPerBlock, 0, ws.streams[0] >> > (
        ws.d_rho, rho_threshold, ws.d_flags, total_pixels
        );
    nvtxRangePop();

    nvtxRangePushA("cub::DeviceSelect");
    // 2. 使用 CUB 库的 Flagged 选择算法，提取掩膜为 1 的像素索引
    thrust::counting_iterator<int> d_in_indices(0);
    size_t temp_storage_bytes = ws.cub_temp_storage_bytes;

    cub::DeviceSelect::Flagged(
        ws.d_cub_temp_storage, temp_storage_bytes,
        d_in_indices, ws.d_flags, ws.d_mask_indices, ws.d_num_selected_out, total_pixels,
        ws.streams[0]
    );
    nvtxRangePop();

    // 3. 异步回传有效像素数量
    CUDA_CHECK(cudaMemcpyAsync(&sel_pixels, ws.d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost, ws.streams[0]));

    if (sel_pixels == 0) {
        std::cerr << "Warning: No pixels satisfy the mask condition!" << std::endl;
    }
    ws.sel_pixels = sel_pixels;
}

// 掩膜数据聚合 Kernel：根据提取的掩膜索引，从原始二维图像中 Gather 出有效像素数据
template <typename T>
__global__ void gather_mask_kernel(const T* img, const int* mask, T* img_sel, int N, int M, int total) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < M && row < N) {
        int mask_pos = mask[col];
        img_sel[row * M + col] = img[row * total + mask_pos];
    }
}

// 单位矩阵生成 Kernel：在显存中初始化 N x N 的单位矩阵
template <typename T>
__global__ void set_identity_kernel(T* d_mat, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int r = idx / N;
        int c = idx % N;
        d_mat[idx] = (r == c) ? static_cast<T>(1.0) : static_cast<T>(0.0);
    }
}

// 生成 Zernike 拟合的伪逆矩阵：基于正规方程 (B^T * B)^-1 * B^T
template <typename T, typename T_phase>
T* zerk_fit_pinv_generate_cuda(PipelineWorkspace<T, T_phase>& ws) {
    int total_pixels = ws.N;    // 原始图像总像素数
    int M = ws.sel_pixels;      // 掩膜筛选后的有效像素数
    int N = ws.N_ORDER;         // Zernike 多项式的阶数（待拟合的系数个数）

    cublasHandle_t cublasH = ws.cublas_handles[0];
    cusolverDnHandle_t cusolverH = ws.cusolver_handle;

    // 注意这里因为一开始不知道M是多少，所以分配的内存空间都是N x total_pixels，因为M < total_pixels
    T* d_B = ws.d_fit_B;        // 掩膜提取后的 Zernike 基矩阵 (N x M)
    T* d_G = ws.d_fit_G;        // Gram 矩阵 G = B^T * B (N x N)
    T* d_G_inv = ws.d_fit_G_inv;// Gram 矩阵的逆 G^-1 (N x N)
    T* d_pinv = ws.d_pinv;      // 最终输出的伪逆矩阵 (N x M)

    // 1. 从全图 Zernike 基中 Gather 出有效像素的数据，构建子矩阵 B
    nvtxRangePushA("gather_mask_kernel");
    dim3 gridDim_gather((M + 15) / 16, (N + 15) / 16);
    dim3 blockDim_gather(16, 16);
    gather_mask_kernel<T> << <gridDim_gather, blockDim_gather, 0, ws.streams[0] >> > (
        ws.d_zernike_basis, ws.d_mask_indices, d_B, N, M, total_pixels
        );
    nvtxRangePop();

    // 2. 计算 Gram 矩阵: G = B^T * B (N x N 的对称正定矩阵)
    nvtxRangePushA("B^T * B");
    const T alpha = 1.0;
    const T beta = 0.0;
    cublasGemm_wrapper<T>(
        cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, M, &alpha, d_B, M, d_B, M, &beta, d_G, N
    );
    nvtxRangePop();

    // 3. 初始化单位矩阵，作为后续求逆的目标矩阵
    nvtxRangePushA("set_identity_kernel");
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    set_identity_kernel<T> << <blocks, threads, 0, ws.streams[0] >> > (d_G_inv, N);
    nvtxRangePop();

    // 4. 查询 Cholesky 分解所需的临时工作空间大小
    nvtxRangePushA("Cholesky bufferSize");
    int lwork = 0;
    cusolverDnPotrf_bufferSize_wrapper<T>(cusolverH, CUBLAS_FILL_MODE_UPPER, N, d_G, N, &lwork);
    nvtxRangePop();

    // 5. Cholesky 分解: 将 G 分解为 U^T * U (原地修改 d_G)
    nvtxRangePushA("G = U^T * U");
    cusolverDnPotrf_wrapper<T>(
        cusolverH, CUBLAS_FILL_MODE_UPPER, N, d_G, N, ws.d_cusolver_work, lwork, ws.d_cusolver_info
    );
    nvtxRangePop();

    // 6. Cholesky 求解: 利用分解结果求解 G * X = I，得到 G^-1 并覆盖 d_G_inv
    nvtxRangePushA("G * X = I get G^-1");
    cusolverDnPotrs_wrapper<T>(
        cusolverH, CUBLAS_FILL_MODE_UPPER, N, N, d_G, N, d_G_inv, N, ws.d_cusolver_info
    );
    nvtxRangePop();

    // 7. 计算伪逆矩阵: pinv = B * G^-1 (即 B * (B^T * B)^-1)
    nvtxRangePushA("pinv = B * G^-1");
    cublasGemm_wrapper<T>(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, N, &alpha, d_B, M, d_G_inv, N, &beta, d_pinv, M
    );
    nvtxRangePop();

    return d_pinv;
}

// 从频域复数数组中提取实部并缩放，用于包络分量A
template <typename T>
__global__ void process_envelope_a_kernel(const typename CuFFTTraits<T>::Complex* in, T* out, int size, T scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx].x * scale;
    }
}

// 从频域复数数组中计算包络分量B（2*模长）和相位角phi
template <typename T>
__global__ void process_envelope_b_and_phi_kernel(const typename CuFFTTraits<T>::Complex* in, T* envelope_b, T* phi, T* rho, int size, T scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T re = in[idx].x * scale;
        T im = in[idx].y * scale;
        envelope_b[idx] = static_cast<T>(2.0) * my_sqrt(re * re + im * im);
        phi[idx] = rho[idx] <= static_cast<T>(1.0) ? my_atan2(im, re) : NAN;
    }
}

// 2D FFT shift：交换四个象限，将零频分量移至频谱中心
template <typename T>
__global__ void fftshift_2d_kernel(typename CuFFTTraits<T>::Complex* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width / 2 && y < height / 2) {
        int x2 = x + width / 2;
        int y2 = y + height / 2;

        // 交换左上与右下象限
        int idx1 = y * width + x;
        int idx3 = y2 * width + x2;
        typename CuFFTTraits<T>::Complex temp1 = data[idx1];
        data[idx1] = data[idx3];
        data[idx3] = temp1;

        // 交换右上与左下象限
        int idx2 = y * width + x2;
        int idx4 = y2 * width + x;
        typename CuFFTTraits<T>::Complex temp2 = data[idx2];
        data[idx2] = data[idx4];
        data[idx4] = temp2;
    }
}

// 实数转复数：虚部置零
template <typename T>
__global__ void real_to_complex_kernel(const T* d_in, typename CuFFTTraits<T>::Complex* d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx].x = d_in[idx];
        d_out[idx].y = static_cast<T>(0.0);
    }
}

// 将任意弧度值归一化到 [-π, π) 区间
template <typename T>
__device__ inline T wrap_device(T v) {
    const T TWO_PI = PhaseConstants<T>::TWO_PI;
    const T PI = PhaseConstants<T>::PI;
    return v - TWO_PI * floor((v + PI) / TWO_PI);
}

// 计算相位图的局部可靠性：通过 3x3 邻域的相位梯度（水平、垂直、两对角线）
// 衡量相位的平滑程度，梯度越小可靠性越高
template <typename T, int TILE_X = 16, int TILE_Y = 16>
__global__ void calc_reliability_kernel(const T* __restrict__ phase, T* __restrict__ reliability, int width, int height) {
    // 带有 1 像素 Halo 边界的二维 Shared Memory Tile [18][18]
    __shared__ T s_phase[TILE_Y + 2][TILE_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * TILE_X + tx;
    const int gy = blockIdx.y * TILE_Y + ty;

    // 内部 Tile 在 Shared Memory 中的映射坐标（偏移 1 以留出 Halo）
    const int sm_x = tx + 1;
    const int sm_y = ty + 1;

    // 边界安全加载 Lambda
    auto load_global = [&](int x, int y) -> T {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            return phase[y * width + x];
        }
        return static_cast<T>(0.0);
        };

    // 1. 加载中心像素到 Shared Memory Tile
    s_phase[sm_y][sm_x] = load_global(gx, gy);

    // 2. 协作加载 4 个方向的 Halo (边界) 像素
    if (tx == 0)          s_phase[sm_y][0] = load_global(gx - 1, gy);
    if (tx == TILE_X - 1) s_phase[sm_y][TILE_X + 1] = load_global(gx + 1, gy);
    if (ty == 0)          s_phase[0][sm_x] = load_global(gx, gy - 1);
    if (ty == TILE_Y - 1) s_phase[TILE_Y + 1][sm_x] = load_global(gx, gy + 1);

    // 3. 协作加载 4 个角落的 Halo 像素
    if (tx == 0 && ty == 0)                      s_phase[0][0] = load_global(gx - 1, gy - 1);
    if (tx == TILE_X - 1 && ty == 0)             s_phase[0][TILE_X + 1] = load_global(gx + 1, gy - 1);
    if (tx == 0 && ty == TILE_Y - 1)             s_phase[TILE_Y + 1][0] = load_global(gx - 1, gy + 1);
    if (tx == TILE_X - 1 && ty == TILE_Y - 1)    s_phase[TILE_Y + 1][TILE_X + 1] = load_global(gx + 1, gy + 1);

    // 块内同步，确保 Tile 数据加载完成
    __syncthreads();

    if (gx >= width || gy >= height) return;
    int idx = gy * width + gx;

    // 边缘像素可靠性设为 0
    if (gx == 0 || gx == width - 1 || gy == 0 || gy == height - 1) {
        reliability[idx] = static_cast<T>(0.0);
        return;
    }

    T val = s_phase[sm_y][sm_x];
    if (isnan(val)) {
        reliability[idx] = static_cast<T>(0.0);
        return;
    }

    // 从 Shared Memory 读取 3x3 邻域像素，消除全局内存重复读取
    // 计算四个方向的相位二阶差分（梯度变化率）
    T dx = wrap_device(s_phase[sm_y][sm_x + 1] - val) - wrap_device(val - s_phase[sm_y][sm_x - 1]);
    T dy = wrap_device(s_phase[sm_y + 1][sm_x] - val) - wrap_device(val - s_phase[sm_y - 1][sm_x]);
    T d1 = wrap_device(s_phase[sm_y + 1][sm_x + 1] - val) - wrap_device(val - s_phase[sm_y - 1][sm_x - 1]);
    T d2 = wrap_device(s_phase[sm_y + 1][sm_x - 1] - val) - wrap_device(val - s_phase[sm_y - 1][sm_x + 1]);

    // gamma 为四方向梯度的 L2 范数，gamma 越小表示相位越平滑
    T gamma = sqrt(dx * dx + dy * dy + d1 * d1 + d2 * d2);

    // 可靠性 = 1/gamma，平滑区域可靠性高，gamma 为 0 时给一个极大值
    reliability[idx] = (gamma == static_cast<T>(0.0)) ? static_cast<T>(1e6) : (static_cast<T>(1.0) / gamma);
}

// ---------------- 32 位 Key 构建核函数 ----------------
// 为每条邻接边生成排序 Key：可靠性越高 Key 越小（降序排列），
// 同时记录两端像素索引和相位跳变整数 k0，供后续 DSU 解包裹使用
template <typename T>
__global__ void build_edges_kernel(
    const T* __restrict__ phase,
    const T* __restrict__ reliability,
    CompactEdge* __restrict__ compact_edges,
    uint32_t* __restrict__ edge_keys,
    int width, int height)
{
    const T TWO_PI = PhaseConstants<T>::TWO_PI;

    // 水平边数 + 垂直边数 = 总边数
    int H = height * (width - 1);
    int V = (height - 1) * width;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= H + V) return;

    int u, v;       // 边的两个端点像素索引
    T w1, w2;       // 两端点的相位值
    T rel;          // 两端点可靠性之和（作为边权重）

    // 水平边：同一行相邻像素
    if (e < H) {
        int y = e / (width - 1);
        int x = e % (width - 1);
        u = y * width + x;
        v = u + 1;
    }
    // 垂直边：同一列相邻像素
    else {
        int t = e - H;
        int y = t / width;
        int x = t % width;
        u = y * width + x;
        v = u + width;
    }

    w1 = phase[u];
    w2 = phase[v];
    rel = reliability[u] + reliability[v];

    // 含 NaN 的无效边：标记为跳过
    if (isnan(w1) || isnan(w2)) {
        compact_edges[e].u = -1;
        compact_edges[e].v = -1;
        compact_edges[e].k0 = 0;
        edge_keys[e] = 0xFFFFFFFFu;
        return;
    }

    // 计算两端相位差对应的 2π 整数倍跳变
    int k0 = (int)round((w1 - w2) / TWO_PI);

    // 生成 32 位可靠性降序 Key：按位取反使高可靠性对应小 Key
    float rel_f = static_cast<float>(rel);
    uint32_t rel_bits = __float_as_uint(rel_f);
    uint32_t key32 = ~rel_bits;

    compact_edges[e].u = u;
    compact_edges[e].v = v;
    compact_edges[e].k0 = k0;
    edge_keys[e] = key32;
}

// 解包裹核函数：将包裹相位加上整数倍 2π 偏移，恢复连续相位
template <typename T>
__global__ void apply_unwrap_kernel(
    const T* __restrict__ wrapped,
    T* __restrict__ unwrapped,
    const int* __restrict__ offset,
    int size)
{
    const T TWO_PI = PhaseConstants<T>::TWO_PI;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    unwrapped[i] = wrapped[i] + static_cast<T>(offset[i]) * TWO_PI;
}

__global__ void dsu_flatten_kernel(DSUNode* __restrict__ dsu_nodes, int* __restrict__ d_offset, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // GPU 线程并行执行路径压缩，计算累计偏移
    int curr = idx;
    int acc_off = 0;

    // 1. 查找根节点并累加 offset
    while (curr != dsu_nodes[curr].parent) {
        acc_off += dsu_nodes[curr].offset;
        curr = dsu_nodes[curr].parent;
    }

    // 2. 将最终累计偏移写入 GPU 内存
    d_offset[idx] = acc_off;
}

// ---------------- 零动态分配 CUB 优化版解包裹 ----------------
// 完整流程：可靠性计算 → 边表构建 → GPU 基数排序 → CPU DSU 并查集 → GPU 回写
template <typename T, typename T_phase>
void unwrap_phase_gpu_hybrid(
    T_phase* d_phase,
    T_phase* d_unwrapped,
    int width, int height,
    PipelineWorkspace<T, T_phase>& ws)
{
    const int N = width * height;
    const int H = height * (width - 1);
    const int V = (height - 1) * width;
    const int E = H + V;

    T_phase* d_reliability = ws.d_unwrap_reliability;
    cudaStream_t stream = ws.streams[0];

    nvtxRangePushA("cal reliability and build edges");
    // 1. 可靠性计算 (指定 Tile 尺寸 16x16)
    {
        constexpr int TILE_X = 16;
        constexpr int TILE_Y = 16;
        dim3 block(TILE_X, TILE_Y);
        dim3 grid((width + TILE_X - 1) / TILE_X, (height + TILE_Y - 1) / TILE_Y);
        calc_reliability_kernel<T_phase, TILE_X, TILE_Y> << <grid, block, 0, stream >> > (d_phase, d_reliability, width, height);
    }

    // 2. 构建 32 位 Key 边表
    {
        int block = 512;
        int grid = (E + 511) / 512;
        build_edges_kernel<T_phase> << <grid, block, 0, stream >> > (d_phase, d_reliability, ws.d_edges, ws.d_keys, width, height);
    }
    nvtxRangePop();

    nvtxRangePushA("sort edges");
    // 3. 原生 CUB 零分配 RadixSort (32-bit Key)：按可靠性降序排列所有边
    cub::DoubleBuffer<uint32_t> d_keys_db(ws.d_keys, ws.d_keys_alt);
    cub::DoubleBuffer<CompactEdge> d_edges_db(ws.d_edges, ws.d_edges_alt);
    size_t temp_storage_bytes = ws.cub_temp_storage_bytes;

    cub::DeviceRadixSort::SortPairs(
        ws.d_cub_temp_storage, temp_storage_bytes,
        d_keys_db, d_edges_db, E,
        0, 32, stream
    );
    nvtxRangePop();

    nvtxRangePushA("cudaMemcpyAsync edges D2H");
    // 4. 异步拷贝排序后的边表至 Pinned Memory 并同步，供 CPU DSU 使用
    CompactEdge* h_edges = ws.h_pinned_edges;
    CUDA_CHECK(cudaMemcpyAsync(h_edges, d_edges_db.Current(), E * sizeof(CompactEdge), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    nvtxRangePushA("DSU");
    // 5. CPU 侧 DSU 并查集：按可靠性从高到低合并连通分量，累积 2π 偏移
    FastDSU dsu;
    dsu.nodes = ws.h_dsu_nodes;
    dsu.init(N);

    DSUNode* __restrict__ dsu_ptr = dsu.nodes;
    int merged_count = 0;

    // --- 早期退出上限 (Early Exit Bound) ---
    // 对于常规网格图，N 个节点最多合并 N - 1 次
    const int max_merges = N - 1;

    for (int e = 0; e < E && merged_count < N - 1; ++e) {
        // 剪枝判断：一旦合并数量达到上限，立即跳出循环，无需再遍历后续低可靠性的边
        if (merged_count >= max_merges) break;

        // 提前预取第 e+24 条边对应的 DSU 节点，以掩盖内存延迟
        PREFETCH_READ(&dsu_ptr[h_edges[std::min(e + 24, E - 1)].u]);
        PREFETCH_READ(&dsu_ptr[h_edges[std::min(e + 24, E - 1)].v]);

        int p1 = h_edges[e].u;
        int p2 = h_edges[e].v;
        if (p1 < 0 || p2 < 0) continue;  // 跳过无效边

        int off1 = 0, off2 = 0;
        int r1 = p1;
        int r2 = p2;

        if (dsu_ptr[p1].parent != p1) {
            r1 = dsu.find(p1, off1);
        }
        if (dsu_ptr[p2].parent != p2) {
            r2 = dsu.find(p2, off2);
        }

        // 两端不在同一连通分量时，按秩合并并记录偏移
        if (r1 != r2) {
            int k = h_edges[e].k0 + off1 - off2;

            // --- 引用解引用与局部变量缓存 ---
            DSUNode& node1 = dsu_ptr[r1];
            DSUNode& node2 = dsu_ptr[r2];

            if (node1.rank < node2.rank) {
                node1.parent = r2;
                node1.offset = -k;
            }
            else {
                node2.parent = r1;
                node2.offset = k;
                if (node1.rank == node2.rank) {
                    node1.rank++;
                }
            }
            ++merged_count;
        }
    }
    nvtxRangePop();

    nvtxRangePushA("Copy DSU Nodes to GPU & Flatten");
    // 将 DSU 节点树直接拷贝回 GPU (大小为 N * sizeof(DSUNode))
    CUDA_CHECK(cudaMemcpyAsync(ws.d_dsu_nodes, dsu_ptr, N * sizeof(DSUNode), cudaMemcpyHostToDevice, stream));

    // 在 GPU 上并行展开路径压缩
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    dsu_flatten_kernel << <grid_size, block_size, 0, stream >> > (ws.d_dsu_nodes, ws.d_unwrap_offset, N);
    nvtxRangePop();

    nvtxRangePushA("apply_unwrap_kernel");
    // GPU 并行执行最终解包裹：unwrapped = wrapped + offset * 2π
    {
        int block = 512;
        int grid = (N + 511) / 512;
        apply_unwrap_kernel << <grid, block, 0, stream >> > (d_phase, d_unwrapped, ws.d_unwrap_offset, N);
    }
    nvtxRangePop();

}

// ---------------- FFT 提取包络与相位 ----------------
// 流程：正向FFT → fftshift → 三通道ROI裁剪 → 反向FFT → 提取包络/相位 → 相位解包裹
template <typename T, typename T_phase>
void envelope_phi_by_fft_2d(const T* d_img, typename CuFFTTraits<T>::Complex* d_img_complex,
    Rect rect_sel[3],
    T* d_envelope_a,
    T* d_envelope_bx,
    T* d_envelope_by,
    T* d_phix,
    T* d_phiy,
    typename CuFFTTraits<T>::Complex* d_fft_a,
    typename CuFFTTraits<T>::Complex* d_fft_bx,
    typename CuFFTTraits<T>::Complex* d_fft_by,
    PipelineWorkspace<T, T_phase>& ws) {

    using ComplexType = typename CuFFTTraits<T>::Complex;
    int size = ws.N;
    int height = ws.H;
    int width = ws.W;
    size_t complex_bytes = size * sizeof(ComplexType);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 block2d(16, 16);
    dim3 grid2d_half((width / 2 + block2d.x - 1) / block2d.x, (height / 2 + block2d.y - 1) / block2d.y);
    nvtxRangePushA("fft_2d");

    cudaStream_t main_stream = ws.streams[0];
    cufftHandle main_plan = ws.fft_plans[0];

    // 实数→复数 → 正向FFT → fftshift 将零频移至中心
    real_to_complex_kernel<T> << <blocksPerGrid, threadsPerBlock, 0, main_stream >> > (d_img, d_img_complex, size);
    CuFFTTraits<T>::execC2C(main_plan, d_img_complex, d_img_complex, CUFFT_FORWARD);
    fftshift_2d_kernel<T> << <grid2d_half, block2d, 0, main_stream >> > (d_img_complex, width, height);

    // 记录FFT完成事件，供下游三个流等待
    CUDA_CHECK(cudaEventRecord(ws.fft_done_event, main_stream));

    // 三通道并行：裁剪ROI → 反向FFT，分别提取零频/±1级频谱
    typename CuFFTTraits<T>::Complex* d_fft_channels[3] = { d_fft_a, d_fft_bx, d_fft_by };

    for (int i = 0; i < 3; ++i) {
        cudaStream_t stream = ws.streams[i];
        cufftHandle plan = ws.fft_plans[i];
        CUDA_CHECK(cudaStreamWaitEvent(stream, ws.fft_done_event, 0));

        cudaMemsetAsync(d_fft_channels[i], 0, complex_bytes, stream);

        int h_roi = rect_sel[i].y_max - rect_sel[i].y_min;
        int w_roi = rect_sel[i].x_max - rect_sel[i].x_min;
        size_t roi_pitch = width * sizeof(ComplexType);

        // 将ROI区域从移位后的频谱中拷贝到各通道缓冲区
        cudaMemcpy2DAsync(
            d_fft_channels[i] + rect_sel[i].y_min * width + rect_sel[i].x_min, roi_pitch,
            ws.d_img_complex + rect_sel[i].y_min * width + rect_sel[i].x_min, roi_pitch,
            w_roi * sizeof(ComplexType), h_roi, cudaMemcpyDeviceToDevice, stream
        );

        // 反fftshift恢复频谱布局 → 反向FFT回到空间域
        fftshift_2d_kernel<T> << <grid2d_half, block2d, 0, stream >> > (d_fft_channels[i], width, height);
        CuFFTTraits<T>::execC2C(plan, d_fft_channels[i], d_fft_channels[i], CUFFT_INVERSE);
    }

    // FFT归一化系数
    T fft_scale = static_cast<T>(1.0) / static_cast<T>(width * height);
    nvtxRangePop();

    nvtxRangePushA("a_b_phi_extrcat");
    // 从复数结果中提取包络（模）和相位（辐角）
    process_envelope_a_kernel<T> << <blocksPerGrid, threadsPerBlock, 0, ws.streams[0] >> > (d_fft_a, d_envelope_a, size, fft_scale);
    process_envelope_b_and_phi_kernel<T> << <blocksPerGrid, threadsPerBlock, 0, ws.streams[1] >> > (d_fft_bx, d_envelope_bx, d_phix, ws.d_rho, size, fft_scale);
    process_envelope_b_and_phi_kernel<T> << <blocksPerGrid, threadsPerBlock, 0, ws.streams[2] >> > (d_fft_by, d_envelope_by, d_phiy, ws.d_rho, size, fft_scale);
    nvtxRangePop();

    // 等待bx/by通道处理完成后再做相位解包裹
    CUDA_CHECK(cudaStreamSynchronize(ws.streams[1]));
    CUDA_CHECK(cudaStreamSynchronize(ws.streams[2]));

    nvtxRangePushA("unwrap_phase");
    unwrap_phase_gpu_hybrid(d_phix, d_phix, width, height, ws);
    unwrap_phase_gpu_hybrid(d_phiy, d_phiy, width, height, ws);
    nvtxRangePop();

}

// ---------------- 图像合成核函数 ----------------
// 根据包络和相位重建图像：img = a + bx*cos(phix) + by*cos(phiy)，rho>=1处置零
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

// ---------------- 掩码 Gather 核函数 ----------------
// 按掩码索引从5个通道中收集有效像素，输出紧凑排列
template <typename T_data, typename T_phase>
__global__ void gather_mask_5channel_kernel(
    const T_data* __restrict__ in0, const T_data* __restrict__ in1, const T_data* __restrict__ in2,
    const T_phase* __restrict__ in3, const T_phase* __restrict__ in4,
    const int* __restrict__ d_mask_indices,
    T_data* __restrict__ d_img_sel_5ch,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        int mask_idx = d_mask_indices[idx];
        d_img_sel_5ch[0 * M + idx] = in0[mask_idx];
        d_img_sel_5ch[1 * M + idx] = in1[mask_idx];
        d_img_sel_5ch[2 * M + idx] = in2[mask_idx];
        d_img_sel_5ch[3 * M + idx] = static_cast<T_phase>(in3[mask_idx]);
        d_img_sel_5ch[4 * M + idx] = static_cast<T_phase>(in4[mask_idx]);
    }
}

// ---------------- 5通道解包核函数 ----------------
// 将紧凑排列的5通道数据拆回独立的a/bx/by/phix/phiy数组
template <typename T_data, typename T_phase>
__global__ void unpack_5channel_kernel(
    const T_data* __restrict__ d_img_re_5ch,
    T_data* __restrict__ out0, T_data* __restrict__ out1, T_data* __restrict__ out2,
    T_phase* __restrict__ out3, T_phase* __restrict__ out4,
    int HW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HW) {
        out0[idx] = d_img_re_5ch[0 * HW + idx];
        out1[idx] = d_img_re_5ch[1 * HW + idx];
        out2[idx] = d_img_re_5ch[2 * HW + idx];
        out3[idx] = static_cast<T_phase>(d_img_re_5ch[3 * HW + idx]);
        out4[idx] = static_cast<T_phase>(d_img_re_5ch[4 * HW + idx]);
    }
}

// double -> float 批量转换
__global__ void double_to_float_kernel(const double* __restrict__ in,
    float* __restrict__ out,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = static_cast<float>(in[idx]);
    }
}

// float -> double 批量转换
__global__ void float_to_double_kernel(const float* __restrict__ in,
    double* __restrict__ out,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = static_cast<double>(in[idx]);
    }
}

// ---------------- Zernike 低通滤波（5通道批量） ----------------
// 流程：掩码Gather → 正向投影(伪逆×数据) → 截断模式 → 反向重建(基×系数) → 解包
template <typename T_data, typename T_phase>
void zernike_low_pass_filter_5channel(
    PipelineWorkspace<T_data, T_phase>& ws,
    T_data* in0, T_data* in1, T_data* in2, T_phase* in3, T_phase* in4,
    T_data* out0, T_data* out1, T_data* out2, T_phase* out3, T_phase* out4,
    T_data* __restrict__ zernike_basis_pinv, T_data* __restrict__ zernike_basis,
    int* d_mask_indices, int H, int W, int n_order, int* h_M,
    int start_mode, int end_mode
) {
    int M = h_M[0];
    if (M <= 0) return;

    cudaStream_t stream = ws.streams[0];
    cublasHandle_t handle = ws.cublas_handles[0];
    cublasSetStream(handle, stream);

    int HW = H * W;

    nvtxRangePushA("gather_mask_5channel_kernel");
    // 按掩码索引收集有效像素，打包为5×M矩阵
    dim3 block_g(256);
    dim3 grid_g(CEIL(M, 256));
    gather_mask_5channel_kernel<T_data, T_phase> << <grid_g, block_g, 0, stream >> > (
        in0, in1, in2, in3, in4,
        d_mask_indices, ws.d_img_sel_5ch, M
        );
    nvtxRangePop();

    nvtxRangePushA("forward");
    // 正向投影：伪逆矩阵 × 数据 → Zernike系数 (n_order × 5)
    const T_data alpha = 1.0, beta = 0.0;
    cublasGemm_wrapper<T_data>(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n_order, 5, M,
        &alpha, zernike_basis_pinv, M,
        ws.d_img_sel_5ch, M,
        &beta, ws.d_zerk_5ch, n_order
    );
    nvtxRangePop();

    nvtxRangePushA("backword");
    // 反向重建：仅使用 [start_mode, end_mode) 范围内的模式
    int num_modes = end_mode - start_mode;
    cublasGemm_wrapper<T_data>(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        HW, 5, num_modes,
        &alpha, zernike_basis + start_mode * HW, HW,
        ws.d_zerk_5ch + start_mode, n_order,
        &beta, ws.d_img_re_5ch, HW
    );
    nvtxRangePop();

    nvtxRangePushA("unpack_5channel_kernel");
    // 将重建结果从紧凑5通道拆回独立数组
    dim3 block_u(256);
    dim3 grid_u(CEIL(HW, 256));
    unpack_5channel_kernel<T_data, T_phase> << <grid_u, block_u, 0, stream >> > (
        ws.d_img_re_5ch, out0, out1, out2, out3, out4, HW
        );
    nvtxRangePop();
}

// ---------------- 残差更新核函数 ----------------
// 更新校正量：corr = fft原始 - fft模型 + 模型（频域残差补偿）
template <typename T_data, typename T_phase>
__global__ void update_residual_5channel_kernel(
    T_data* corr0, const T_data* fft0, const T_data* fft_model0, const T_data* model0,
    T_data* corr1, const T_data* fft1, const T_data* fft_model1, const T_data* model1,
    T_data* corr2, const T_data* fft2, const T_data* fft_model2, const T_data* model2,
    T_phase* corr3, const T_phase* fft3, const T_phase* fft_model3, const T_phase* model3,
    T_phase* corr4, const T_phase* fft4, const T_phase* fft_model4, const T_phase* model4,
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        corr0[idx] = fft0[idx] - fft_model0[idx] + model0[idx];
        corr1[idx] = fft1[idx] - fft_model1[idx] + model1[idx];
        corr2[idx] = fft2[idx] - fft_model2[idx] + model2[idx];
        corr3[idx] = fft3[idx] - fft_model3[idx] + model3[idx];
        corr4[idx] = fft4[idx] - fft_model4[idx] + model4[idx];
    }
}

// ---------------- 迭代FFT主流程 ----------------
// 流程：初始FFT提取 → 滤波+重建+残差更新 → 迭代精修
template <typename T_data, typename T_phase>
void iter_fft_2d_pipeline(
    Rect rect_sel[3],
    T_data* d_pinv_data, T_data* d_zernike_basis_data,
    T_data* d_rho, T_data rho_range_fit, int* d_mask_indices, int M, int n_order, int iter_num,
    int& h_M, int start_mode, int end_mode,
    T_data* d_a_corr, T_data* d_bx_corr, T_data* d_by_corr,
    T_phase* d_phix_corr, T_phase* d_phiy_corr,
    PipelineWorkspace<T_data, T_phase>& ws
) {
    int total_pixels = M * M;
    dim3 cor_threads(16, 16);
    dim3 cor_blocks(CEIL(M, 16), CEIL(M, 16));
    dim3 threads(256);
    dim3 blocks(CEIL(total_pixels, 256));

    cudaStream_t main_stream = ws.streams[0];

    // 对5通道执行Zernike低通滤波，输出模型分量
    auto launch_5channel_filter = [&](T_data* a_in, T_data* bx_in, T_data* by_in,
        T_phase* phix_in, T_phase* phiy_in) {
            zernike_low_pass_filter_5channel<T_data, T_phase>(
                ws,
                a_in, bx_in, by_in, phix_in, phiy_in,
                ws.d_a_model, ws.d_bx_model, ws.d_by_model, ws.d_phix_model, ws.d_phiy_model,
                d_pinv_data, d_zernike_basis_data, d_mask_indices, M, M, n_order, &h_M,
                start_mode, end_mode
            );
        };

    // 更新5通道残差：corr = fft原始 - fft模型 + 模型
    auto launch_residual_update = [&]() {
        update_residual_5channel_kernel<T_data, T_phase> << <blocks, threads, 0, main_stream >> > (
            d_a_corr, ws.d_a_fft, ws.d_a_fft_model, ws.d_a_model,
            d_bx_corr, ws.d_bx_fft, ws.d_bx_fft_model, ws.d_bx_model,
            d_by_corr, ws.d_by_fft, ws.d_by_fft_model, ws.d_by_model,
            d_phix_corr, ws.d_phix_fft, ws.d_phix_fft_model, ws.d_phix_model,
            d_phiy_corr, ws.d_phiy_fft, ws.d_phiy_fft_model, ws.d_phiy_model,
            total_pixels
            );
        };

    // 1. 初始 2D FFT 提取
    nvtxRangePushA("1st fft extract");
    envelope_phi_by_fft_2d<T_data, T_phase>(
        ws.d_img, ws.d_img_complex, rect_sel,
        ws.d_a_fft, ws.d_bx_fft, ws.d_by_fft, ws.d_phix_fft, ws.d_phiy_fft,
        ws.d_fft_a, ws.d_fft_bx, ws.d_fft_by, ws
    );
    nvtxRangePop();

    // 2. 初始滤波与残差更新
    nvtxRangePushA("zernike filter");
    launch_5channel_filter(ws.d_a_fft, ws.d_bx_fft, ws.d_by_fft, ws.d_phix_fft, ws.d_phiy_fft);
    nvtxRangePop();

    // 用模型分量合成图像，再对合成图像做FFT，得到模型的频域表示
    img_generate_2d << <cor_blocks, cor_threads, 0, main_stream >> > (
        ws.d_img, d_rho, ws.d_a_model, ws.d_bx_model, ws.d_by_model, ws.d_phix_model, ws.d_phiy_model, M
        );

    nvtxRangePushA("2nd fft extract");
    envelope_phi_by_fft_2d<T_data, T_phase>(
        ws.d_img, ws.d_img_complex, rect_sel,
        ws.d_a_fft_model, ws.d_bx_fft_model, ws.d_by_fft_model, ws.d_phix_fft_model, ws.d_phiy_fft_model,
        ws.d_fft_a, ws.d_fft_bx, ws.d_fft_by, ws
    );
    nvtxRangePop();

    // 计算初始残差
    launch_residual_update();

    // 3. 迭代精修：滤波→合成→FFT→残差，逐步逼近
    for (int i = 0; i < iter_num; i++) {
        // 对当前残差做Zernike低通滤波
        launch_5channel_filter(d_a_corr, d_bx_corr, d_by_corr, d_phix_corr, d_phiy_corr);

        // 用滤波结果合成新图像
        img_generate_2d << <cor_blocks, cor_threads, 0, main_stream >> > (
            ws.d_img, d_rho, ws.d_a_model, ws.d_bx_model, ws.d_by_model, ws.d_phix_model, ws.d_phiy_model, M
            );

        nvtxRangePushA("iter_loop_step");
        // 对合成图像做FFT提取频域分量
        envelope_phi_by_fft_2d<T_data, T_phase>(
            ws.d_img, ws.d_img_complex, rect_sel,
            ws.d_a_fft_model, ws.d_bx_fft_model, ws.d_by_fft_model, ws.d_phix_fft_model, ws.d_phiy_fft_model,
            ws.d_fft_a, ws.d_fft_bx, ws.d_fft_by, ws
        );

        // 更新残差，进入下一轮迭代
        launch_residual_update();
        nvtxRangePop();
    }
}

int main(void) {
    nvtxRangePushA("MainLoop");
    auto start1 = std::chrono::high_resolution_clock::now();

    using T_data = double;
    using T_phase = double;
    int M = 2048;
    int N_ORDER = 64;
    bool is_norm = false;
    int iter_num = 5;

    const T_data spot_radius_nm = 9e6;
    const T_data pixel_size_nm = 10e3;
    T_data cmos_radius = M * pixel_size_nm / 2.0;
    T_data rho_range_fit = 0.995;
    T_data ddx = 27e3;
    T_data ddy = 27e3;
    T_data ddz = 9e6 / std::tan(std::asin(1.35 / 4.0));
    T_data scale_factor = 2 * M_PI / 193.368;

    nvtxRangePushA("PipelineWorkspace Init");
    PipelineWorkspace<T_data, T_data> ws;
    ws.init(M, M, N_ORDER);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    auto start2 = std::chrono::high_resolution_clock::now();

    // 1. 坐标与系统误差生成
    dim3 threads(16, 16);
    dim3 blocks(CEIL(M, 16), CEIL(M, 16));
    T_data norm_val = static_cast<T_data>(cmos_radius / spot_radius_nm);
    T_data step = static_cast<T_data>(2.0) * norm_val / (M - 1);

    cudaStream_t main_stream = ws.streams[0];

    nvtxRangePushA("Preheat");
    system_error_generate<T_data, T_phase> << <blocks, threads, 0, main_stream >> > (norm_val, step, cmos_radius, ddx, ddy, ddz, scale_factor, M, ws);
    nvtxRangePop();

    nvtxRangePushA("system_error_generate");
    system_error_generate<T_data, T_phase> << <blocks, threads, 0, main_stream >> > (norm_val, step, cmos_radius, ddx, ddy, ddz, scale_factor, M, ws);
    nvtxRangePop();

    auto start3 = std::chrono::high_resolution_clock::now();

    // 2. Zernike 基底与伪逆生成
    int zerk_threads = 512;
    int zerk_blocks = (M * M + zerk_threads - 1) / zerk_threads;

    nvtxRangePushA("create_zernike_params");
    create_zernike_params<T_data, T_phase> << <1, N_ORDER, 0, main_stream >> > (ws);
    nvtxRangePop();

    // 分配动态 Shared Memory 尺寸 ---
    size_t shared_mem_size = N_ORDER * sizeof(ZernikeOrderParam<T_data>);

    nvtxRangePushA("generate_zernike_basis");
    generate_zernike_basis_kernel<T_data, T_phase> << <zerk_blocks, zerk_threads, shared_mem_size, main_stream >> > (ws, is_norm);
    nvtxRangePop();

    nvtxRangePushA("generate_mask");
    generate_mask<T_data, T_phase>(ws, rho_range_fit, ws.sel_pixels);
    nvtxRangePop();

    nvtxRangePushA("zerk_pinv_generate");
    T_data* d_pinv = zerk_fit_pinv_generate_cuda<T_data, T_phase>(ws);
    nvtxRangePop();

    auto start4 = std::chrono::high_resolution_clock::now();

    // 3. 迭代 FFT 提取主流程
    int start_mode = 0;
    int end_mode = 64;
    Rect rect_sel[3];
    rect_sel[0] = { 974, 1074, 974, 1074 };
    rect_sel[1] = { 974, 1074, 1074, 1174 };
    rect_sel[2] = { 1074, 1174, 974, 1074 };

    nvtxRangePushA("iter_fft_2d_pipeline");
    iter_fft_2d_pipeline<T_data, T_phase>(
        rect_sel,
        d_pinv, ws.d_zernike_basis,
        ws.d_rho, rho_range_fit, ws.d_mask_indices, M, N_ORDER, iter_num,
        ws.sel_pixels, start_mode, end_mode,
        ws.d_a_corr, ws.d_bx_corr, ws.d_by_corr,
        ws.d_phix, ws.d_phiy,
        ws
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    auto start5 = std::chrono::high_resolution_clock::now();

    //std::vector<T_phase> phix(ws.N);
    //cudaMemcpy(phix.data(), ws.d_phix, ws.N * sizeof(T_phase), cudaMemcpyDeviceToHost);
    //save_raw<T_phase>("phix.bin", phix);
    //std::vector<T_phase> phiy(ws.N);
    //cudaMemcpy(phiy.data(), ws.d_phiy, ws.N * sizeof(T_phase), cudaMemcpyDeviceToHost);
    //save_raw<T_phase>("phiy.bin", phiy);

    nvtxRangePushA("PipelineWorkspace destroy");
    ws.destroy();
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePop();
    auto start6 = std::chrono::high_resolution_clock::now();

    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start1);
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(start3 - start2);
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(start4 - start3);
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(start5 - start4);
    auto duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(start6 - start5);
    auto duration6 = std::chrono::duration_cast<std::chrono::milliseconds>(start6 - start1);

    std::cout << "memory allocate: " << duration1.count() << " ms" << std::endl;
    std::cout << "phase image generate: " << duration2.count() << " ms" << std::endl;
    std::cout << "zernike basis generate: " << duration3.count() << " ms" << std::endl;
    std::cout << "iter FFT extract phase: " << duration4.count() << " ms" << std::endl;
    std::cout << "memory release: " << duration5.count() << " ms" << std::endl;
    std::cout << "total times: " << duration6.count() << " ms" << std::endl;

    return 0;
}