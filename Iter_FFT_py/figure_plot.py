#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/5/17 9:19
# @Author :
# @File : figure_plot.py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

def plt_imshow(img, color_clip=0.0, title='', cmap='jet'):
    plt.figure()
    v_min = np.percentile(img, 50*color_clip)
    v_max = np.percentile(img, 50*(2 - color_clip))
    plt.imshow(img, vmin=v_min, vmax=v_max, cmap=cmap)
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.show(block=True)

def plt_1d(data, title=''):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.show(block=True)
