#include <cuda_runtime.h>

#define M_PI 3.14159265358979323846

template <typename T>
__global__ void system_error_generate_kernel(T cmos_radius, T ddx, T ddy, T ddz, T scale_factor, T* phi_x, T* phi_y, int M) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= M || col >= M) return;

	T xx = static_cast<T>((-1.0 + 2.0 * col / (M - 1)) * cmos_radius);
	T yy = static_cast<T>((-1.0 + 2.0 * row / (M - 1)) * cmos_radius);

	T xx_r2 = xx * xx;
	T yy_r2 = yy * yy;
	T ddz_r2 = ddz * ddz;

	// 分别计算三个距离
	T dist_x = sqrt((xx - ddx) * (xx - ddx) + yy_r2 + ddz_r2);
	T dist_y = sqrt(xx_r2 + (yy - ddy) * (yy - ddy) + ddz_r2);
	T dist_ref = sqrt(xx_r2 + yy_r2 + ddz_r2);

	// 【数值优化】：利用 (A - B) / (sqrt(A) + sqrt(B)) 规避大数相减的消去误差
	// 分子为 ddx * ddx - 2.0 * xx * ddx，消去了巨大的 ddz^2
	T diff_x = (ddx * ddx - static_cast<T>(2.0) * xx * ddx) / (dist_x + dist_ref);
	T diff_y = (ddy * ddy - static_cast<T>(2.0) * yy * ddy) / (dist_y + dist_ref);

	phi_x[row * M + col] = diff_x * scale_factor;
	phi_y[row * M + col] = diff_y * scale_factor;

}

template <typename T>
void system_error_generate(
	T cmos_radius, T ddx, T ddy, T ddz,
	T scale_factor,
	T* phi_x, T* phi_y, int M) {
	dim3 threads(16, 16);
	dim3 blocks((M + 16 - 1) / 16, (M + 16 - 1) / 16);
	system_error_generate_kernel<T> << <blocks, threads >> > (
		cmos_radius, ddx, ddy, ddz, scale_factor, phi_x, phi_y, M);
}