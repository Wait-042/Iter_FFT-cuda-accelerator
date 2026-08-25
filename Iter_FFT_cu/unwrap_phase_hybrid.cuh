#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <chrono>

#if defined(__GNUC__) || defined(__clang__)
    #define PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 1)
#elif defined(_MSC_VER)
    #include <xmmintrin.h>
    #define PREFETCH_READ(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
    #define PREFETCH_READ(addr)
#endif

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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

// ---------------- 设备端 wrap ----------------
template <typename T>
__device__ inline T wrap_device(T v) {
    const T TWO_PI = PhaseConstants<T>::TWO_PI;
    const T PI = PhaseConstants<T>::PI;
    return v - TWO_PI * floor((v + PI) / TWO_PI);
}

// ---------------- 紧凑结构体：仅包含 u,v,k0 （12字节） ----------------
struct CompactEdge {
    int u, v, k0;
};

// ---------------- 核函数：可靠性 ----------------
template <typename T>
__global__ void calc_reliability_kernel(
    const T* __restrict__ phase,
    T* __restrict__ reliability,
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

    T val = phase[idx];
    if (isnan(val)) { reliability[idx] = 0.0; return; }

    T dx = wrap_device(phase[idx + 1] - val) - wrap_device(val - phase[idx - 1]);
    T dy = wrap_device(phase[idx + width] - val) - wrap_device(val - phase[idx - width]);
    T d1 = wrap_device(phase[idx + width + 1] - val) - wrap_device(val - phase[idx - width - 1]);
    T d2 = wrap_device(phase[idx + width - 1] - val) - wrap_device(val - phase[idx - width + 1]);

    T gamma = sqrt(dx * dx + dy * dy + d1 * d1 + d2 * d2);
    reliability[idx] = (gamma == 0.0) ? 1e6 : (1.0 / gamma);
}

// ---------------- 核函数：构建边 ----------------
template <typename T>
__global__ void build_edges_kernel(
    const T* __restrict__ phase,
    const T* __restrict__ reliability,
    CompactEdge* __restrict__ compact_edges,
    uint64_t* __restrict__ edge_keys,
    int width, int height)
{
    const T TWO_PI = PhaseConstants<T>::TWO_PI;

    int H = height * (width - 1);
    int V = (height - 1) * width;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= H + V) return;

    int u, v;
    T w1, w2;
    T rel;

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
        compact_edges[e].u = -1;
        compact_edges[e].v = -1;
        compact_edges[e].k0 = 0;
        edge_keys[e] = ~0ULL;   // 最大值，排序后置于末尾
        return;
    }

    int k0 = (int)round((w1 - w2) / TWO_PI);

    // 生成32位可靠性降序键
    float f = (float)rel;
    uint32_t key32 = __float_as_uint(f);
    key32 = ~key32;

    // 64位唯一键
    uint64_t key64 = ((uint64_t)key32 << 32) | (uint32_t)e;

    compact_edges[e].u = u;
    compact_edges[e].v = v;
    compact_edges[e].k0 = k0;
    edge_keys[e] = key64;
}

// ---------------- 核函数：输出解包裹相位 ----------------
template <typename T>
__global__ void apply_unwrap_kernel(
    const T* __restrict__ wrapped,
    T* __restrict__ unwrapped,
    const int* __restrict__ parent,
    const int* __restrict__ offset,
    int size)
{
    const T TWO_PI = PhaseConstants<T>::TWO_PI;

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

// ---------------- 16 字节缓存对齐的 DSU 节点 ----------------
struct alignas(16) DSUNode {
    int parent;
    int offset;
    int rank;
    int padding; // 补齐至 16 字节，刚好填满一个 Cache Line 段
};

struct FastDSU {
    std::vector<DSUNode> nodes;

    FastDSU(int n) : nodes(n) {
        for (int i = 0; i < n; ++i) {
            nodes[i] = {i, 0, 0, 0};
        }
    }

    inline int find(int i, int& total_offset) {
        DSUNode* __restrict__ pnodes = nodes.data();
        int curr = i;
        int acc_off = 0;

        while (curr != pnodes[curr].parent) {
            acc_off += pnodes[curr].offset;
            curr = pnodes[curr].parent;
        }
        int root = curr;
        total_offset = acc_off;

        curr = i;
        int path_off = 0;
        while (curr != root) {
            int next = pnodes[curr].parent;
            int old_off = pnodes[curr].offset;
            pnodes[curr].parent = root;
            pnodes[curr].offset = acc_off - path_off;
            path_off += old_off;
            curr = next;
        }
        return root;
    }
};

// ---------------- 主函数：混合 GPU 解包裹（完全优化版） ----------------
template <typename T>
void unwrap_phase_gpu_hybrid(
    const T* h_phase,
    T* h_unwrapped,
    int width, int height)
{
    const int N = width * height;
    const int H = height * (width - 1);
    const int V = (height - 1) * width;
    const int E = H + V;

    // ---------- 设备内存分配 ----------
    T* d_phase, * d_reliability;
    CompactEdge* d_edges;
    uint64_t* d_keys;
    CUDA_CHECK(cudaMalloc(&d_phase, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_reliability, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_edges, E * sizeof(CompactEdge)));
    CUDA_CHECK(cudaMalloc(&d_keys, E * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_phase, h_phase, N * sizeof(T), cudaMemcpyHostToDevice));

    // ---------- 计时变量 ----------
    cudaEvent_t ev_start, ev_after_reliability, ev_after_build, ev_after_sort, ev_after_copy;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_after_reliability));
    CUDA_CHECK(cudaEventCreate(&ev_after_build));
    CUDA_CHECK(cudaEventCreate(&ev_after_sort));
    CUDA_CHECK(cudaEventCreate(&ev_after_copy));
    CUDA_CHECK(cudaEventRecord(ev_start));

    // 1. 可靠性
    {
        dim3 block(32, 32);
        dim3 grid((width + 31) / 32, (height + 31) / 32);
        calc_reliability_kernel<<<grid, block>>>(d_phase, d_reliability, width, height);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(ev_after_reliability));

    // 2. 构建边
    {
        int block = 256;
        int grid = (E + 255) / 256;
        build_edges_kernel<<<grid, block>>>(d_phase, d_reliability, d_edges, d_keys, width, height);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(ev_after_build));

    // 3. 排序：改用非稳定排序 thrust::sort_by_key（利用键唯一性提升 GPU 排序效率）
    thrust::device_ptr<uint64_t> d_keys_ptr(d_keys);
    thrust::device_ptr<CompactEdge> d_edges_ptr(d_edges);
    thrust::sort_by_key(d_keys_ptr, d_keys_ptr + E, d_edges_ptr);
    CUDA_CHECK(cudaEventRecord(ev_after_sort));

    // 4. 拷贝排序后的紧凑结构体到主机（固定内存）
    CompactEdge* h_edges;
    CUDA_CHECK(cudaMallocHost(&h_edges, E * sizeof(CompactEdge)));
    CUDA_CHECK(cudaMemcpy(h_edges, d_edges, E * sizeof(CompactEdge), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_after_copy));

    // ---------- 测量 GPU 各阶段 ----------
    float ms_rel, ms_build, ms_sort, ms_copy;
    CUDA_CHECK(cudaEventSynchronize(ev_after_copy));
    CUDA_CHECK(cudaEventElapsedTime(&ms_rel, ev_start, ev_after_reliability));
    CUDA_CHECK(cudaEventElapsedTime(&ms_build, ev_after_reliability, ev_after_build));
    CUDA_CHECK(cudaEventElapsedTime(&ms_sort, ev_after_build, ev_after_sort));
    CUDA_CHECK(cudaEventElapsedTime(&ms_copy, ev_after_sort, ev_after_copy));
    //printf("GPU Reliability: %.2f ms\n", ms_rel);
    //printf("GPU Build edges: %.2f ms\n", ms_build);
    //printf("GPU Sort: %.2f ms\n", ms_sort);
    //printf("GPU Copy out (pinned, compact): %.2f ms\n", ms_copy);

    // ---------- 5. CPU 合并（极限优化版） ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    FastDSU dsu(N);
    DSUNode* __restrict__ dsu_ptr = dsu.nodes.data();
    int merged_count = 0;

    for (int e = 0; e < E && merged_count < N - 1; ++e) {
        // 软件预取：提前 16 条边加载数据至 L1 Cache
        PREFETCH_READ(&dsu_ptr[h_edges[std::min(e + 16, E - 1)].u]);
        PREFETCH_READ(&dsu_ptr[h_edges[std::min(e + 16, E - 1)].v]);

        int p1 = h_edges[e].u;
        int p2 = h_edges[e].v;
        if (p1 < 0 || p2 < 0) continue;

        int off1 = 0, off2 = 0;
        int r1 = dsu_ptr[p1].parent;
        int r2 = dsu_ptr[p2].parent;

        // Fast Path：如果节点本身即为根节点，直接命中，省去 find 内部循环
        if (r1 == p1) {
            off1 = 0;
        } else {
            r1 = dsu.find(p1, off1);
        }

        if (r2 == p2) {
            off2 = 0;
        } else {
            r2 = dsu.find(p2, off2);
        }

        if (r1 != r2) {
            int k = h_edges[e].k0 + off1 - off2;
            if (dsu_ptr[r1].rank < dsu_ptr[r2].rank) {
                dsu_ptr[r1].parent = r2;
                dsu_ptr[r1].offset = -k;
            } else {
                dsu_ptr[r2].parent = r1;
                dsu_ptr[r2].offset = k;
                if (dsu_ptr[r1].rank == dsu_ptr[r2].rank) {
                    dsu_ptr[r1].rank++;
                }
            }
            ++merged_count;
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    //printf("CPU Merge (optimized): %lld ms\n", (long long)cpu_ms);

    // 释放 pinned 内存
    cudaFreeHost(h_edges);

    // ---------- 6. 输出 ----------
    std::vector<int> h_parent(N), h_offset(N);
    for (int i = 0; i < N; ++i) {
        h_parent[i] = dsu.nodes[i].parent;
        h_offset[i] = dsu.nodes[i].offset;
    }

    int* d_parent, * d_offset;
    CUDA_CHECK(cudaMalloc(&d_parent, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_parent, h_parent.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, h_offset.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    T* d_unwrapped;
    CUDA_CHECK(cudaMalloc(&d_unwrapped, N * sizeof(T)));
    {
        int block = 256;
        int grid = (N + 255) / 256;
        apply_unwrap_kernel<<<grid, block>>>(d_phase, d_unwrapped, d_parent, d_offset, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(h_unwrapped, d_unwrapped, N * sizeof(T), cudaMemcpyDeviceToHost));

    // 清理资源
    cudaFree(d_phase); cudaFree(d_reliability); cudaFree(d_edges); cudaFree(d_keys);
    cudaFree(d_parent); cudaFree(d_offset); cudaFree(d_unwrapped);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_after_reliability);
    cudaEventDestroy(ev_after_build); cudaEventDestroy(ev_after_sort); cudaEventDestroy(ev_after_copy);
}