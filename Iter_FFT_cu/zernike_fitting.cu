#include "zernike_fitting.cuh"

// 1. 普通阶乘函数实现
__host__ double factorial(int k) {
    double res = 1.0;
    for (int i = 2; i <= k; ++i) res *= i;
    return res;
}

// 2. 泽尼克索引阶数计算实现
__host__ std::vector<int2> zernike_indices(int n_order) {
    std::vector<int2> indices;
    for (int i = 1; i <= n_order; ++i) {
        int d = static_cast<int>(std::floor(std::sqrt(i - 1))) + 1;
        int m = (((d * d - i) & 1) != 0) ? (-d * d + i - 1) / 2 : (d * d - i) / 2;
        int n = 2 * (d - 1) - std::abs(m);
        indices.push_back(make_int2(m, n));
    }
    return indices;
}

// 3. 非模板 Kernel 实现
__global__ void generate_zernike_basis_kernel(
    double* __restrict__ rho,
    double* __restrict__ theta,
    double* __restrict__ basis,
    int2* __restrict__ mn_indices,
    int H, int W, int n_order, bool is_norm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = H * W;
    if (idx >= total_pixels) return;

    double r = rho[idx];
    double t = theta[idx];

    // 超出单位圆（孔径）直接置零
    if (r > 1.0) {
        for (int o = 0; o < n_order; ++o) {
            basis[o * total_pixels + idx] = 0.0;
        }
        return;
    }

    double r2 = r * r;

    for (int o = 0; o < n_order; ++o) {
        int m = mn_indices[o].x;
        int n = mn_indices[o].y;
        int abs_m = abs(m);
        int max_s = (n - abs_m) / 2;

        double radial = 0.0;

        // 特殊情况：圆心 (r == 0) 特殊处理
        if (r == 0.0) {
            if (m == 0) {
                radial = ((n / 2) & 1) ? -1.0 : 1.0;
            }
            else {
                radial = 0.0;
            }
        }
        else {
            int k1 = (n + abs_m) / 2;
            int k2 = (n - abs_m) / 2;

            double term_denom = 1.0;
            for (int k = 1; k <= k1; ++k) term_denom *= k;
            for (int k = 1; k <= k2; ++k) term_denom *= k;

            double c0 = 1.0;
            for (int k = 1; k <= n; ++k) c0 *= k;
            c0 /= term_denom;

            double current_term = c0 * pow(r, n);
            radial = current_term;

            for (int s = 0; s < max_s; ++s) {
                double ratio = -static_cast<double>((k1 - s) * (k2 - s)) / ((s + 1) * (n - s) * r2);
                current_term *= ratio;
                radial += current_term;
            }
        }

        double val = 0.0;
        if (m > 0)       val = radial * cos(m * t);
        else if (m < 0)  val = radial * sin(-m * t);
        else             val = radial;

        if (is_norm) {
            double norm = (m == 0) ? sqrt(n + 1.0) : sqrt(2.0 * (n + 1.0));
            val *= norm;
        }

        basis[o * total_pixels + idx] = val;
    }
}