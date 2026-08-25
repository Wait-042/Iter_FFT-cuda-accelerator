#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/5/17 10:49
# @Author :
# @File : iter_fft.py
import time

import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase

import config

if config.CUPY:
    import cupy as np
    from cupyx.scipy.fft import fft2, ifft2, fftshift, ifftshift
else:
    import numpy as np
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
from image_generate import system_error_generate
from zernike_fitting import zernike_basis_generate, zerk_fit_pinv_generate, zernike_low_pass_filter
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号


def coordinate_generate():
    """
    生成归一化极坐标 (rho, theta)，用于 Zernike 多项式计算
    :return: rho (归一化径向坐标), theta (极角)
    """
    num_pixel = config.NUM_PIXEL
    spot_radius_nm = config.SPOT_RADIUS_NM
    pixel_size_nm = config.PIXEL_SIZE_NM
    cmos_radius = num_pixel * pixel_size_nm / 2  # CMOS 靶面物理半径 (nm)

    # 生成归一化到 [-1, 1] 的物理坐标网格
    x_norm = np.linspace(-1, 1, num_pixel) * cmos_radius / spot_radius_nm
    xx_norm, yy_norm = np.meshgrid(x_norm, x_norm)

    # 计算极坐标
    rho = np.sqrt(xx_norm ** 2 + yy_norm ** 2)
    theta = np.arctan2(yy_norm, xx_norm)
    return rho, theta


def img_filter(img, mask_sel):
    """
    通过矩形掩膜在频域/空域进行滤波
    :param img: 输入图像
    :param mask_sel: 掩膜边界 [row_start, row_end, col_start, col_end]
    :return: 滤波后的图像
    """
    kernel = np.zeros_like(img)
    kernel[mask_sel[0]:mask_sel[1], mask_sel[2]:mask_sel[3]] = 1
    img_f = img * kernel
    return img_f


def envelope_phi_by_fft_1d(img, rect_sel, direction='x'):
    """
    基于 FFT 提取干涉条纹的背景光强(a)、调制振幅(b)和包裹相位(phi)
    :param img: 干涉条纹图
    :param rect_sel: 包含 0频 和 +1级 频域滤波窗口的列表
    :return: envelope_a, envelope_b, phi
    """
    img_fft = fftshift(fft2(img))

    # 提取 0频 分量和 +1级 分量
    img_fft_a = img_filter(img_fft, rect_sel[0])
    if direction == 'x':
        img_fft_b = img_filter(img_fft, rect_sel[1])
    elif direction == 'y':
        img_fft_b = img_filter(img_fft, rect_sel[2])
    else:
        raise Exception('direction incorrect')

    # 逆变换提取各分量
    envelope_a = np.real(ifft2(ifftshift(img_fft_a)))
    envelope_b = 2 * np.abs(ifft2(ifftshift(img_fft_b)))
    phi = np.angle(ifft2(ifftshift(img_fft_b)))

    # 相位解包裹（兼容 GPU/CPU 环境）
    if config.CUPY:
        phi = np.asarray(unwrap_phase(np.asnumpy(phi)))
    else:
        phi = unwrap_phase(phi)
    return envelope_a, envelope_b, phi


def envelope_phi_by_fft_2d(img, rect_sel):
    """
    基于 FFT 提取干涉条纹的背景光强(a)、调制振幅(b)和包裹相位(phi)
    :param img: 干涉条纹图
    :param rect_sel: 包含 0频 和 +1级 频域滤波窗口的列表
    :return: envelope_a, envelope_b, phi
    """
    img_fft = fftshift(fft2(img))

    # 提取 0频 分量和 +1级 分量
    img_fft_a = img_filter(img_fft, rect_sel[0])
    img_fft_bx = img_filter(img_fft, rect_sel[1])
    img_fft_by = img_filter(img_fft, rect_sel[2])

    # 逆变换提取各分量
    envelope_a = np.real(ifft2(ifftshift(img_fft_a)))
    envelope_bx = 2 * np.abs(ifft2(ifftshift(img_fft_bx)))
    envelope_by = 2 * np.abs(ifft2(ifftshift(img_fft_by)))
    phi_x = np.angle(ifft2(ifftshift(img_fft_bx)))
    phi_y = np.angle(ifft2(ifftshift(img_fft_by)))

    # 相位解包裹（兼容 GPU/CPU 环境）
    if config.CUPY:
        phi_x = np.asarray(unwrap_phase(np.asnumpy(phi_x)))
        phi_y = np.asarray(unwrap_phase(np.asnumpy(phi_y)))
    else:
        phi_x = unwrap_phase(phi_x)
        phi_y = unwrap_phase(phi_y)
    return envelope_a, envelope_bx, envelope_by, phi_x, phi_y


def iter_fft_1d(img, rect_sel, zernike_basis, zernike_basis_pinv, n_order, rho, rho_range, iter_num):
    """
    迭代 FFT 算法：结合 Zernike 低通滤波，消除频域滤波带来的边缘伪影和高频噪声
    :param img: 原始干涉条纹图
    :param rect_sel: 频域滤波窗口
    :param zernike_basis: Zernike 基底矩阵
    :param zernike_basis_pinv: Zernike 拟合伪逆矩阵
    :param n_order: Zernike 拟合阶数
    :param rho: 归一化径向坐标
    :param rho_range: 有效拟合区域半径
    :param iter_num: 迭代次数
    :return: 校正后的 a_corr, b_corr, phi_corr
    """
    # 1. 初始 FFT 提取
    a_fft, b_fft, phi_fft = envelope_phi_by_fft_1d(img, rect_sel)

    # 2. Zernike 低通滤波（平滑处理）
    a_model = zernike_low_pass_filter(a_fft, zernike_basis_pinv, zernike_basis, n_order, rho)
    b_model = zernike_low_pass_filter(b_fft, zernike_basis_pinv, zernike_basis, n_order, rho)
    phi_model = zernike_low_pass_filter(phi_fft, zernike_basis_pinv, zernike_basis, n_order, rho)

    # 3. 生成虚拟干涉图并计算误差
    img_vir = a_model + b_model * np.cos(phi_model)
    img_vir[rho > rho_range] = 0.
    a_fft_model, b_fft_model, phi_fft_model = envelope_phi_by_fft_1d(img_vir, rect_sel)

    delta_a = a_fft_model - a_model
    delta_b = b_fft_model - b_model
    delta_phi = phi_fft_model - phi_model

    # 4. 初始校正
    a_corr = a_fft - delta_a
    b_corr = b_fft - delta_b
    phi_corr = phi_fft - delta_phi

    # 5. 迭代校正循环
    for i in range(iter_num):
        a_model = zernike_low_pass_filter(a_corr, zernike_basis_pinv, zernike_basis, n_order, rho)
        b_model = zernike_low_pass_filter(b_corr, zernike_basis_pinv, zernike_basis, n_order, rho)
        phi_model = zernike_low_pass_filter(phi_corr, zernike_basis_pinv, zernike_basis, n_order, rho)

        img_vir = a_model + b_model * np.cos(phi_model)
        img_vir[rho > rho_range] = 0.

        a_fft_model, b_fft_model, phi_fft_model = envelope_phi_by_fft_1d(img_vir, rect_sel)

        # 更新误差并校正
        delta_a = a_fft_model - a_model
        delta_b = b_fft_model - b_model
        delta_phi = phi_fft_model - phi_model

        a_corr = a_fft - delta_a
        b_corr = b_fft - delta_b
        phi_corr = phi_fft - delta_phi

    return a_corr, b_corr, phi_corr


def iter_fft_2d(img, rect_sel, zernike_basis, zernike_basis_pinv, n_order, rho, rho_range, iter_num):
    """
    迭代 FFT 算法：结合 Zernike 低通滤波，消除频域滤波带来的边缘伪影和高频噪声
    :param img: 原始干涉条纹图
    :param rect_sel: 频域滤波窗口
    :param zernike_basis: Zernike 基底矩阵
    :param zernike_basis_pinv: Zernike 拟合伪逆矩阵
    :param n_order: Zernike 拟合阶数
    :param rho: 归一化径向坐标
    :param rho_range: 有效拟合区域半径
    :param iter_num: 迭代次数
    :return: 校正后的 a_corr, b_corr, phi_corr
    """
    # 1. 初始 FFT 提取
    a_fft, bx_fft, by_fft, phix_fft, phiy_fft = envelope_phi_by_fft_2d(img, rect_sel)

    # 2. Zernike 低通滤波（平滑处理）
    a_model = zernike_low_pass_filter(a_fft, zernike_basis_pinv, zernike_basis, n_order, rho)
    bx_model = zernike_low_pass_filter(bx_fft, zernike_basis_pinv, zernike_basis, n_order, rho)
    by_model = zernike_low_pass_filter(by_fft, zernike_basis_pinv, zernike_basis, n_order, rho)
    phix_model = zernike_low_pass_filter(phix_fft, zernike_basis_pinv, zernike_basis, n_order, rho)
    phiy_model = zernike_low_pass_filter(phiy_fft, zernike_basis_pinv, zernike_basis, n_order, rho)

    # 3. 生成虚拟干涉图并计算误差
    img_vir = a_model + bx_model * np.cos(phix_model) + by_model * np.cos(phiy_model)
    img_vir[rho > rho_range] = 0.

    a_fft_model, bx_fft_model, by_fft_model, phix_fft_model, phiy_fft_model = envelope_phi_by_fft_2d(img_vir, rect_sel)

    delta_a = a_fft_model - a_model
    delta_bx = bx_fft_model - bx_model
    delta_by = by_fft_model - by_model
    delta_phix = phix_fft_model - phix_model
    delta_phiy = phiy_fft_model - phiy_model

    # 4. 初始校正
    a_corr = a_fft - delta_a
    bx_corr = bx_fft - delta_bx
    by_corr = by_fft - delta_by
    phix_corr = phix_fft - delta_phix
    phiy_corr = phiy_fft - delta_phiy

    # 5. 迭代校正循环
    for i in range(iter_num):
        a_model = zernike_low_pass_filter(a_corr, zernike_basis_pinv, zernike_basis, n_order, rho)
        bx_model = zernike_low_pass_filter(bx_corr, zernike_basis_pinv, zernike_basis, n_order, rho)
        by_model = zernike_low_pass_filter(by_corr, zernike_basis_pinv, zernike_basis, n_order, rho)
        phix_model = zernike_low_pass_filter(phix_corr, zernike_basis_pinv, zernike_basis, n_order, rho)
        phiy_model = zernike_low_pass_filter(phiy_corr, zernike_basis_pinv, zernike_basis, n_order, rho)

        img_vir = a_model + bx_model * np.cos(phix_model) + by_model * np.cos(phiy_model)
        img_vir[rho > rho_range] = 0.

        a_fft_model, bx_fft_model, by_fft_model, phix_fft_model, phiy_fft_model = envelope_phi_by_fft_2d(img_vir,
                                                                                                         rect_sel)

        # 更新误差并校正
        delta_a = a_fft_model - a_model
        delta_bx = bx_fft_model - bx_model
        delta_by = by_fft_model - by_model
        delta_phix = phix_fft_model - phix_model
        delta_phiy = phiy_fft_model - phiy_model

        a_corr = a_fft - delta_a
        bx_corr = bx_fft - delta_bx
        by_corr = by_fft - delta_by
        phix_corr = phix_fft - delta_phix
        phiy_corr = phiy_fft - delta_phiy

    return a_corr, bx_corr, by_corr, phix_corr, phiy_corr


def phase_shift_solve(img):
    phase_warp = np.arctan2((img[3] - img[1]), (img[0] - img[2]))
    if config.CUPY:
        phase_unwarp = np.asarray(unwrap_phase(np.asnumpy(phase_warp)))
    else:
        phase_unwarp = unwrap_phase(phase_warp)
    return phase_unwarp


def python_cuda_diff_plt(aa_corr, bx_corr, by_corr, phix_corr, phiy_corr, dtype, fraction=0.05, pad=0.05):
    aa_corr_cuda = np.fromfile(r".\data\aa_corr_float64.bin", dtype=dtype).reshape(num_pixel, num_pixel)
    bx_corr_cuda = np.fromfile(r".\data\bx_corr_float64.bin", dtype=dtype).reshape(num_pixel, num_pixel)
    by_corr_cuda = np.fromfile(r".\data\by_corr_float64.bin", dtype=dtype).reshape(num_pixel, num_pixel)
    phix_corr_cuda = np.fromfile(r".\data\phix_corr_float64.bin", dtype=dtype).reshape(num_pixel, num_pixel)
    phiy_corr_cuda = np.fromfile(r".\data\phiy_corr_float64.bin", dtype=dtype).reshape(num_pixel, num_pixel)

    # Python & CUDA 迭代FFT结果对比
    delta_aa_py_cuda = aa_corr_cuda - aa_corr
    delta_bx_py_cuda = bx_corr_cuda - bx_corr
    delta_by_py_cuda = by_corr_cuda - by_corr
    delta_phix_py_cuda = phix_corr_cuda - phix_corr
    delta_phiy_py_cuda = phiy_corr_cuda - phiy_corr
    delta_phix_py_cuda[rho > rho_range_fit] = np.nan
    delta_phiy_py_cuda[rho > rho_range_fit] = np.nan
    delta_phix_py_cuda -= np.nanmean(delta_phix_py_cuda)
    delta_phiy_py_cuda -= np.nanmean(delta_phiy_py_cuda)

    if config.CUPY:
        delta_aa_py_cuda = np.asnumpy(delta_aa_py_cuda)
        delta_bx_py_cuda = np.asnumpy(delta_bx_py_cuda)
        delta_by_py_cuda = np.asnumpy(delta_by_py_cuda)
        delta_phix_py_cuda = np.asnumpy(delta_phix_py_cuda)
        delta_phiy_py_cuda = np.asnumpy(delta_phiy_py_cuda)

    plt.figure(figsize=(20, 6))
    plt.subplot(151)
    plt.imshow(delta_aa_py_cuda, cmap='jet')
    plt.title('迭代FFT Python & CUDA 包络a差异')
    plt.colorbar(fraction=fraction, pad=pad)

    plt.subplot(152)
    plt.imshow(delta_bx_py_cuda, cmap='jet')
    plt.title('迭代FFT Python & CUDA 包络bx差异')
    plt.colorbar(fraction=fraction, pad=pad)

    plt.subplot(153)
    plt.imshow(delta_by_py_cuda, cmap='jet')
    plt.title('迭代FFT Python & CUDA 包络by差异')
    plt.colorbar(fraction=fraction, pad=pad)

    plt.subplot(154)
    plt.imshow(delta_phix_py_cuda, cmap='jet')
    plt.title('迭代FFT Python & CUDA 相位phix差异')
    plt.colorbar(fraction=fraction, pad=pad)

    plt.subplot(155)
    plt.imshow(delta_phiy_py_cuda, cmap='jet')
    plt.title('迭代FFT Python & CUDA 相位phiy差异')
    plt.colorbar(fraction=fraction, pad=pad)

    plt.suptitle('迭代FFT Python & CUDA 提取包络相位结果差异')
    plt.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    start1 = time.time()

    # 算法超参数配置
    n_order = config.N_ORDER  # Zernike 拟合阶数
    rho_range = 1.0  # 有效孔径范围
    iter_num = 5  # 迭代校正次数
    win_len = 50  # 频域滤波窗口半宽
    phase_step = 1  #相移步数
    num_pixel = config.NUM_PIXEL  # 图像分辨率
    rho_range_fit = config.RHO_RANGE_FIT

    # 定义频域滤波窗口坐标 [row_start, row_end, col_start, col_end]
    rect_sel = [[1024 - win_len, 1024 + win_len, 1024 - win_len, 1024 + win_len],  # 0频窗口
                [1024 - win_len, 1024 + win_len, 1124 - win_len, 1124 + win_len],  # +1级 X 窗口
                [1124 - win_len, 1124 + win_len, 1024 - win_len, 1024 + win_len],  # +1级 Y 窗口
                [1124 - win_len, 1124 + win_len, 1124 - win_len, 1124 + win_len]]

    # 生成模拟系统误差和干涉条纹图
    phi_x, phi_y = system_error_generate()
    a = np.ones_like(phi_x)
    b = np.ones_like(phi_x)
    img_1d = np.zeros((phase_step, num_pixel, num_pixel))
    img_2d = np.zeros((phase_step, num_pixel, num_pixel))
    for i in range(phase_step):
        img_1d[i] = a + b * np.cos(phi_x + i * np.pi / 2)
        img_2d[i] = a + b * np.cos(phi_x + i * np.pi / 2) + b * np.cos(phi_y + i * np.pi / 2)

    # 生成坐标并裁剪单位圆外区域
    rho, theta = coordinate_generate()
    img_1d[:, rho > rho_range] = 0.

    img_2d[:, rho > rho_range] = 0.

    start2 = time.time()

    # 预计算 Zernike 基底和伪逆矩阵
    zernike_basis = zernike_basis_generate(rho, theta, n_order, is_norm_zernike=False)
    start3 = time.time()

    # zernike_basis = zernike_basis_recursion_generate(n_order, rho, theta, is_norm_zernike=False)
    zernike_basis_pinv = zerk_fit_pinv_generate(zernike_basis, rho)
    start4 = time.time()

    aa_corr = np.zeros_like(img_1d)
    bx_corr = np.zeros_like(img_1d)
    by_corr = np.zeros_like(img_1d)
    phix_corr = np.zeros_like(img_1d)
    phiy_corr = np.zeros_like(img_1d)
    for i in range(phase_step):
        aa_corr[i], bx_corr[i], by_corr[i], phix_corr[i], phiy_corr[i] = iter_fft_2d(img_2d[i], rect_sel, zernike_basis,
                                                                                     zernike_basis_pinv, n_order, rho,
                                                                                     rho_range, iter_num)

    start5 = time.time()

    # 根据 FFT 提取包络相位重构图像
    img_2d_x_iter_fft = np.zeros((phase_step, num_pixel, num_pixel))
    img_2d_y_iter_fft = np.zeros((phase_step, num_pixel, num_pixel))

    for i in range(phase_step):
        img_2d_x_iter_fft[i] = aa_corr[i] + bx_corr[i] * np.cos(phix_corr[i])
        img_2d_y_iter_fft[i] = aa_corr[i] + by_corr[i] * np.cos(phiy_corr[i])

    # 相移解算提取相位
    if phase_step == 4:
        phi_x_iter_fft_unwarp = phase_shift_solve(img_2d_x_iter_fft)
        phi_y_iter_fft_unwarp = phase_shift_solve(img_2d_y_iter_fft)
    start6 = time.time()

    python_cuda_diff_plt(aa_corr[0], bx_corr[0], by_corr[0], phix_corr[0], phiy_corr[0], dtype=np.float64)

    print('相位、图像生成耗时/s: ', start2 - start1)
    print('zernike基生成耗时/s: ', start3 - start2)
    print('zernike基伪逆生成耗时/s: ', start4 - start3)
    print('2d条纹普通FFT提取相位耗时/s: ', start5 - start4)
    print('相移解算耗时/s: ', start6 - start5)
    print('整体耗时/s: ', start6 - start1)
