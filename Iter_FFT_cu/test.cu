#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <fstream>

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <chrono>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

// ---------------- 设备端 wrap ----------------
__device__ inline double wrap_device(double v) {
    return v - TWO_PI * floor((v + PI) / TWO_PI);
}

// ---------------- 核函数：可靠性 ----------------
__global__ void calc_reliability_kernel(
    const double* __restrict__ phase,
    double* __restrict__ reliability,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        reliability[idx] = 0.0;
        return;
    }

    double val = phase[idx];
    if (isnan(val)) { reliability[idx] = 0.0; return; }

    double dx = wrap_device(phase[idx + 1] - val) - wrap_device(val - phase[idx - 1]);
    double dy = wrap_device(phase[idx + width] - val) - wrap_device(val - phase[idx - width]);
    double d1 = wrap_device(phase[idx + width + 1] - val) - wrap_device(val - phase[idx - width - 1]);
    double d2 = wrap_device(phase[idx + width - 1] - val) - wrap_device(val - phase[idx - width + 1]);

    double gamma = sqrt(dx * dx + dy * dy + d1 * d1 + d2 * d2);
    reliability[idx] = (gamma == 0.0) ? 1e6 : (1.0 / gamma);
}

// ---------------- 核函数：构建边 ----------------
__global__ void build_edges_kernel(
    const double* __restrict__ phase,
    const double* __restrict__ reliability,
    int* __restrict__ edge_u,
    int* __restrict__ edge_v,
    int* __restrict__ edge_k0,
    uint64_t* __restrict__ edge_key,
    int width, int height)
{
    int H = height * (width - 1);
    int V = (height - 1) * width;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= H + V) return;

    int u, v;
    double w1, w2;
    double rel;

    if (e < H) {
        int y = e / (width - 1);
        int x = e % (width - 1);
        u = y * width + x;
        v = u + 1;
    }
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

    if (isnan(w1) || isnan(w2)) {
        edge_u[e] = -1;
        edge_v[e] = -1;
        edge_k0[e] = 0;
        edge_key[e] = ~0ULL;
        return;
    }

    int k0 = (int)round((w1 - w2) / TWO_PI);

    float f = (float)rel;
    uint32_t key32;
    memcpy(&key32, &f, sizeof(uint32_t));
    key32 = ~key32;

    uint64_t key64 = ((uint64_t)key32 << 32) | (uint32_t)e;

    edge_u[e] = u;
    edge_v[e] = v;
    edge_k0[e] = k0;
    edge_key[e] = key64;
}

// ---------------- 核函数：输出解包裹相位 ----------------
__global__ void apply_unwrap_kernel(
    const double* __restrict__ wrapped,
    double* __restrict__ unwrapped,
    const int* __restrict__ parent,
    const int* __restrict__ offset,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    int curr = i;
    int acc_off = 0;
    while (curr != parent[curr]) {
        acc_off += offset[curr];
        curr = parent[curr];
    }
    unwrapped[i] = wrapped[i] + acc_off * TWO_PI;
}

// ---------------- 主机端 FastDSU（复用您的代码） ----------------
struct FastDSU {
    std::vector<int> parent;
    std::vector<int> offset;
    std::vector<uint8_t> rank;

    FastDSU(int n) : parent(n), offset(n, 0), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int i, int& total_offset) {
        int curr = i;
        int acc_off = 0;
        while (curr != parent[curr]) {
            acc_off += offset[curr];
            curr = parent[curr];
        }
        int root = curr;
        total_offset = acc_off;

        curr = i;
        int path_off = 0;
        while (curr != root) {
            int next = parent[curr];
            int old_off = offset[curr];
            parent[curr] = root;
            offset[curr] = acc_off - path_off;
            path_off += old_off;
            curr = next;
        }
        return root;
    }
};

// ---------------- 主函数：混合 GPU 解包裹 ----------------
void unwrap_phase_gpu_hybrid(
    const double* h_phase,
    double* h_unwrapped,
    int width, int height)
{
    const int N = width * height;
    const int H = height * (width - 1);
    const int V = (height - 1) * width;
    const int E = H + V;

    // 设备内存
    double* d_phase, * d_reliability;
    int* d_u, * d_v, * d_k0;
    uint64_t* d_key;

    CUDA_CHECK(cudaMalloc(&d_phase, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_reliability, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_k0, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_key, E * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_phase, h_phase, N * sizeof(double), cudaMemcpyHostToDevice));

    // 1. 可靠性
    {
        dim3 block(32, 32);
        dim3 grid((width + 31) / 32, (height + 31) / 32);
        calc_reliability_kernel << <grid, block >> > (d_phase, d_reliability, width, height);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. 构建边
    {
        int block = 256;
        int grid = (E + 255) / 256;
        build_edges_kernel << <grid, block >> > (d_phase, d_reliability, d_u, d_v, d_k0, d_key, width, height);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. 排序
    thrust::device_vector<uint64_t> d_keys(d_key, d_key + E);
    thrust::device_vector<int> d_u_vec(d_u, d_u + E);
    thrust::device_vector<int> d_v_vec(d_v, d_v + E);
    thrust::device_vector<int> d_k0_vec(d_k0, d_k0 + E);

    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(),
        thrust::make_zip_iterator(thrust::make_tuple(d_u_vec.begin(), d_v_vec.begin(), d_k0_vec.begin())));

    // 4. 回传主机
    std::vector<int> h_u(E), h_v(E), h_k0(E);
    thrust::copy(d_u_vec.begin(), d_u_vec.end(), h_u.begin());
    thrust::copy(d_v_vec.begin(), d_v_vec.end(), h_v.begin());
    thrust::copy(d_k0_vec.begin(), d_k0_vec.end(), h_k0.begin());

    // 5. CPU 合并（FastDSU）
    FastDSU dsu(N);
    for (int e = 0; e < E; ++e) {
        int p1 = h_u[e], p2 = h_v[e];
        if (p1 < 0 || p2 < 0) continue;
        int off1 = 0, off2 = 0;
        int r1 = dsu.find(p1, off1);
        int r2 = dsu.find(p2, off2);
        if (r1 != r2) {
            int k = h_k0[e] + off1 - off2;
            if (dsu.rank[r1] < dsu.rank[r2]) {
                dsu.parent[r1] = r2;
                dsu.offset[r1] = -k;
            }
            else {
                dsu.parent[r2] = r1;
                dsu.offset[r2] = k;
                if (dsu.rank[r1] == dsu.rank[r2]) dsu.rank[r1]++;
            }
        }
    }

    // 6. 输出（可将 parent/offset 拷贝到 GPU，并行输出）
    // 这里直接复制到 GPU 并调用核函数
    int* d_parent, * d_offset;
    CUDA_CHECK(cudaMalloc(&d_parent, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_parent, dsu.parent.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, dsu.offset.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    double* d_unwrapped;
    CUDA_CHECK(cudaMalloc(&d_unwrapped, N * sizeof(double)));
    {
        int block = 256;
        int grid = (N + 255) / 256;
        apply_unwrap_kernel << <grid, block >> > (d_phase, d_unwrapped, d_parent, d_offset, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(h_unwrapped, d_unwrapped, N * sizeof(double), cudaMemcpyDeviceToHost));

    // 清理
    cudaFree(d_phase); cudaFree(d_reliability);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_k0); cudaFree(d_key);
    cudaFree(d_parent); cudaFree(d_offset); cudaFree(d_unwrapped);
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

// ---------------- 测试 ----------------
//int main() {
//    const int W = 2048, H = 2048;
//    const int N = W * H;
//    std::vector<double> phase(N), unwrapped(N);
//    // 生成测试数据（与您之前的相同）
//    for (int i = 0; i < H; ++i)
//        for (int j = 0; j < W; ++j) {
//            double true_phase = 0.02 * (i + j);
//            phase[i * W + j] = fmod(true_phase + PI, TWO_PI) - PI;
//        }
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start); cudaEventCreate(&stop);
//    cudaEventRecord(start);
//    unwrap_phase_gpu_hybrid(phase.data(), unwrapped.data(), W, H);
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float ms;
//    cudaEventElapsedTime(&ms, start, stop);
//    printf("Total time: %f ms\n", ms);
//    save_raw<double>("./data/unwrapped.bin", unwrapped);
//
//    // 输出前几个值检查
//    for (int i = 0; i < 5; ++i)
//        printf("wrapped[%d]=%f  unwrapped[%d]=%f\n", i, phase[i], i, unwrapped[i]);
//
//    cudaEventDestroy(start); cudaEventDestroy(stop);
//    return 0;
//}