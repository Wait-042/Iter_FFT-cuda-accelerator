#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/5/17 9:19
# @Author :
# @File : image_generate.py
import config

if config.CUPY:
    import cupy as np
else:
    import numpy as np


def system_error_generate():
    """生成系统误差引起的相位分布（X/Y方向）"""
    # 从配置中提取光学与传感器参数
    num_pixel = config.NUM_PIXEL
    na = config.NA
    magnification = config.MAGNIFICATION
    spot_radius_nm = config.SPOT_RADIUS_NM
    pixel_size_nm = config.PIXEL_SIZE_NM
    cmos_radius = num_pixel * pixel_size_nm / 2  # CMOS 靶面物理半径 (nm)
    ddz_nm = spot_radius_nm / np.tan(np.arcsin(na / magnification))  # 参考点离焦量 (nm)
    ddx_nm = config.DDX_NM  # X方向横向偏移
    ddy_nm = config.DDY_NM  # Y方向横向偏移
    wavelength = config.WAVE_LENGTH_NM

    scale_factor = 2 * np.pi / wavelength  # 光程差(OPD)到相位的转换系数

    # 构建 CMOS 靶面二维物理坐标网格
    x = np.linspace(-1, 1, num_pixel) * cmos_radius
    xx, yy = np.meshgrid(x, x)

    # 定义参考点及X/Y方向偏移点的三维坐标
    pos_ref = [0, 0, ddz_nm]
    pos_ref_x = [ddx_nm, 0, ddz_nm]
    pos_ref_y = [0, ddy_nm, ddz_nm]

    # 计算 X/Y 偏移点相对于中心参考点的光程差 (OPD)
    opd_x = np.sqrt((xx - pos_ref_x[0]) ** 2 + (yy - pos_ref_x[1]) ** 2 + (0 - pos_ref_x[2]) ** 2) - np.sqrt(
        (xx - pos_ref[0]) ** 2 + (yy - pos_ref[1]) ** 2 + (0 - pos_ref[2]) ** 2)
    opd_y = np.sqrt((xx - pos_ref_y[0]) ** 2 + (yy - pos_ref_y[1]) ** 2 + (0 - pos_ref_y[2]) ** 2) - np.sqrt(
        (xx - pos_ref[0]) ** 2 + (yy - pos_ref[1]) ** 2 + (0 - pos_ref[2]) ** 2)

    # 将光程差转换为相位
    phi_x = opd_x * scale_factor
    phi_y = opd_y * scale_factor

    return phi_x, phi_y