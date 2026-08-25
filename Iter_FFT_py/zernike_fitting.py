#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/5/18 20:30
# @Author :
# @File : zernike_fitting.py
import math
import numexpr
import numpy
import config
if config.CUPY:
    import cupy as np
else:
    import numpy as np

def zernike_indices(n_order):
    """
    生成 Zernike 多项式的 (m, n) 索引序列
    :param n_order: 需要生成的多项式项数
    :return: 包含 (方位角频率m, 径向频率n) 的列表
    """
    indices = []
    for i in range(1, n_order + 1):
        # 计算当前项所在的径向阶数 d
        d = int(np.floor(np.sqrt(i - 1))) + 1

        # 根据 Noll 索引规则计算方位角频率 m
        if (d ** 2 - i) & 1:
            m = (- d ** 2 + i - 1) // 2
        else:
            m = (d ** 2 - i) // 2

        # 计算径向频率 n
        n = int(2 * (d - 1) - np.abs(m))

        indices.append((m, n))
    return indices


def zernike_formula_generate(n_order, is_norm_zernike=False):
    """
    生成 Zernike 多项式的字符串表达式（用于 numexpr 加速计算）
    :param n_order: 多项式项数
    :param is_norm_zernike: 是否使用归一化 Zernike 多项式
    :return: 字典 {索引: 字符串表达式}
    """
    indices = zernike_indices(n_order)
    zernike_formulas = {}

    for i in range(1, n_order + 1):
        r_terms = []
        m, n = indices[i - 1]

        # 计算径向多项式 R_n^m(rho) 的各项系数并拼接
        for s in range((n - abs(m)) // 2 + 1):
            temp = ((-1) ** s * math.factorial(n - s)) / (
                    math.factorial(s) * math.factorial((n + abs(m)) // 2 - s) * math.factorial(
                (n - abs(m)) // 2 - s))

            if n - 2 * s:
                r_terms.append(f"{int(temp)}*rho**{n - 2 * s}" if temp != 1 else f"rho**{n - 2 * s}")
            else:
                r_terms.append(f"{int(temp)}" if temp != 1 else f"rho**{n - 2 * s}")

        r_capital = " + ".join(r_terms).replace("+ -", "- ")

        # 根据 m 的正负拼接角度部分 (cos 或 sin) 及归一化系数
        if m > 0:
            z_terms_norm = f"sqrt({2 * (n + 1)})*({r_capital})*cos({m}*theta)"
            z_terms = f"({r_capital})*cos({m}*theta)"
        elif m < 0:
            z_terms_norm = f"sqrt({2 * (n + 1)})*({r_capital})*sin({-m}*theta)"
            z_terms = f"({r_capital})*sin({-m}*theta)"
        else:
            z_terms_norm = f"sqrt({(n + 1)})*({r_capital})"
            z_terms = f"({r_capital})"

        zernike_formulas[i] = z_terms_norm if is_norm_zernike else z_terms

    return zernike_formulas


def zernike_basis_recursion_generate(n_order, rho, theta, is_norm_zernike=False):
    """
    通过直接公式计算生成 Zernike 多项式基底矩阵（纯 NumPy 实现）
    :param n_order: 多项式项数
    :param rho: 归一化径向坐标
    :param theta: 极角坐标
    :param is_norm_zernike: 是否归一化
    :return: shape=(n_order, H, W) 的基底矩阵
    """
    indices = zernike_indices(n_order)
    zernike_basis = np.zeros((n_order, rho.shape[0], rho.shape[1]))

    for i in range(1, n_order + 1):
        r_capital = 0.
        m, n = indices[i - 1]

        # 计算径向多项式 R_n^m(rho)
        for s in range((n - abs(m)) // 2 + 1):
            temp = ((-1) ** s * math.factorial(n - s)) / (
                    math.factorial(s) * math.factorial((n + abs(m)) // 2 - s) * math.factorial(
                (n - abs(m)) // 2 - s))
            r_capital += temp * rho ** (n - 2 * s)

        # 组合径向与角度部分
        if m > 0:
            z_terms_norm = np.sqrt(2 * (n + 1)) * r_capital * np.cos(m * theta)
            z_terms = r_capital * np.cos(m * theta)
        elif m < 0:
            z_terms_norm = np.sqrt(2 * (n + 1)) * r_capital * np.sin(-m * theta)
            z_terms = r_capital * np.sin(-m * theta)
        else:
            z_terms_norm = np.sqrt(n + 1) * r_capital
            z_terms = r_capital

        zernike_basis[i - 1] = z_terms_norm if is_norm_zernike else z_terms

    return zernike_basis


def zernike_basis_generate(rho, theta, n_order, is_norm_zernike=False):
    """
    使用 numexpr 加速生成 Zernike 多项式基底矩阵
    :param rho: 归一化径向坐标
    :param theta: 极角坐标
    :param n_order: 多项式项数
    :param is_norm_zernike: 是否归一化
    :return: shape=(n_order, H, W) 的基底矩阵
    """
    zernike_formulas = zernike_formula_generate(n_order, is_norm_zernike)
    zernike_basis = numpy.zeros((n_order, rho.shape[0], rho.shape[1]))

    # 若配置了 GPU 加速，将数据转回 CPU 以支持 numexpr
    if config.CUPY:
        rho = np.asnumpy(rho)
        theta = np.asnumpy(theta)

    # 逐阶计算 Zernike 基底
    for i in range(n_order):
        zernike_basis[i] = numexpr.evaluate(zernike_formulas[i + 1],
                                            local_dict={'rho': rho,
                                                        'theta': theta,
                                                        'sqrt': numpy.sqrt,
                                                        'cos': numpy.cos,
                                                        'sin': numpy.sin
                                                        })

    # 将单位圆外的区域设为 NaN
    zernike_basis[:, rho > 1.0] = np.nan
    zernike_basis = np.asarray(zernike_basis)

    return zernike_basis


def zerk_fit_pinv_generate(zernike_basis, rho):
    """
    预计算 Zernike 拟合的伪逆矩阵（提升拟合速度）
    :param zernike_basis: Zernike 基底矩阵 (N, H, W)
    :param rho: 归一化径向坐标 (H, W)
    :return: 伪逆矩阵
    """
    # 提取有效拟合区域（单位圆内）
    mask = np.where(rho <= config.RHO_RANGE_FIT)
    zernike_basis_sel = zernike_basis[:, mask[0], mask[1]].transpose(1, 0)

    # 计算 Moore-Penrose 伪逆: (B^T B)^-1 B^T
    zernike_basis_pinv = np.dot(np.linalg.pinv(np.dot(zernike_basis_sel.T, zernike_basis_sel)), zernike_basis_sel.T)
    return zernike_basis_pinv


def zerk_fit(img, zernike_basis_pinv, rho):
    """
    利用预计算的伪逆矩阵进行 Zernike 系数拟合
    :param img: 待拟合的波前/图像 (H, W)
    :param zernike_basis_pinv: 预计算的伪逆矩阵
    :param rho: 归一化径向坐标 (H, W)
    :return: Zernike 系数向量
    """
    mask = np.where(rho <= config.RHO_RANGE_FIT)
    img_sel = img[mask]
    # 矩阵乘法获取各阶系数
    zerk = np.dot(zernike_basis_pinv, img_sel)
    return zerk


def wavefront_recover(zerk, zernike_basis, start=0, end=64):
    """
    根据 Zernike 系数和基底矩阵重建波前
    :param zerk: Zernike 系数向量
    :param zernike_basis: Zernike 基底矩阵 (N, H, W)
    :param start: 重建的起始阶数
    :param end: 重建的结束阶数
    :return: 重建的波前矩阵 (H, W)
    """
    # 使用爱因斯坦求和约定进行系数与基底的加权叠加
    wavefront_re = np.einsum('i, ijk -> jk', zerk[start:end], zernike_basis[start:end])
    return wavefront_re


def zernike_low_pass_filter(img, zernike_basis_pinv, zernike_basis, n_order, rho):
    """
    基于 Zernike 多项式的低通滤波器（拟合后重建，滤除高频噪声）
    :param img: 原始图像/波前
    :param zernike_basis_pinv: 预计算的伪逆矩阵
    :param zernike_basis: Zernike 基底矩阵
    :param n_order: 保留的低频阶数
    :param rho: 归一化径向坐标
    :return: 滤波后的平滑图像/波前
    """
    # 1. 拟合 Zernike 系数
    zerk = zerk_fit(img, zernike_basis_pinv, rho)
    # 2. 仅使用前 n_order 阶系数重建波前（截断高频）
    img_re = wavefront_recover(zerk, zernike_basis, 0, n_order)
    # 3. 将 NaN 替换为 0
    img_re = np.nan_to_num(img_re)
    return img_re



if __name__ == '__main__':
    res = zernike_indices(64)
    zernike_formulas = zernike_formula_generate(64)
    for key, value in zernike_formulas.items():
        # print(value)
        data = numexpr.evaluate(value,
                         local_dict={'rho': 1.1,
                                     'theta': 0.3,
                                     'sqrt': np.sqrt,
                                     'cos': np.cos,
                                     'sin': np.sin
                                     })
        print(data)
    pass