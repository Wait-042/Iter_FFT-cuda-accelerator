# 基于 迭代FFT 的条纹图参数提取公式推导

## 1. 信号模型与欧拉展开

原始空域信号模型为：

$$
I(x, y) = a(x, y) + b(x, y) \cos[\phi(x, y)]
$$

利用欧拉公式 $\cos(\theta) = \frac{e^{j\theta} + e^{-j\theta}}{2}$，将其展开为指数形式：

$$
I(x, y) = a(x, y) + \frac{b(x, y)}{2} e^{j\phi(x, y)} + \frac{b(x, y)}{2} e^{-j\phi(x, y)}
$$

---

## 2. 傅里叶变换（频谱域表示）

对 $I(x, y)$ 进行二维傅里叶变换，利用频移性质，得到频谱 $F(u, v)$：

$$
F(u, v) = \mathcal{F}\{a\} + \frac{1}{2}\mathcal{F}\{b \cdot e^{j\phi}\} + \frac{1}{2}\mathcal{F}\{b \cdot e^{-j\phi}\}
$$

在频谱中，这三项分别对应：
- **0 频分量（直流项）**：$F_0(u, v) = \mathcal{F}\{a\}$
- **+1 级频谱**：$F_{+1}(u, v) = \frac{1}{2}\mathcal{F}\{b \cdot e^{j\phi}\}$
- **-1 级频谱**：$F_{-1}(u, v) = \frac{1}{2}\mathcal{F}\{b \cdot e^{-j\phi}\}$

---

## 3. 频域滤波与逆变换提取

通过频域滤波（如带通滤波或窗口掩膜）分离出各分量，并进行逆傅里叶变换（IFFT）。

### 3.1 提取背景光强 $a(x, y)$

对 0 频分量直接进行逆变换：

$$
a(x, y) = \mathcal{F}^{-1}\{F_0(u, v)\}
$$

### 3.2 提取 +1 级分量复信号

对 +1 级频谱进行逆变换，得到解析复信号 $c(x, y)$：

$$
c(x, y) = \mathcal{F}^{-1}\{F_{+1}(u, v)\} = \frac{b(x, y)}{2} e^{j\phi(x, y)}
$$

---

## 4. 从复信号中提取振幅 $b$ 和相位 $\phi$

设提取到的复信号为 $c(x, y) = \text{Re}[c] + j \cdot \text{Im}[c]$，根据复数极坐标表示：

### 4.1 提取调制振幅 $b(x, y)$

取复信号的模并乘以 2：

$$
b(x, y) = 2 \cdot |c(x, y)| = 2 \sqrt{\text{Re}[c(x, y)]^2 + \text{Im}[c(x, y)]^2}
$$

### 4.2 提取相位 $\phi(x, y)$

取复信号的辐角（使用四象限反正切函数避免相位跳变）：

$$
\phi(x, y) = \arg[c(x, y)] = \text{atan2}\Big(\text{Im}[c(x, y)], \ \text{Re}[c(x, y)]\Big)
$$

> **⚠️ 注意：** $\text{atan2}(y, x)$ 返回的相位值范围为 $(-\pi, \pi]$，若原始相位 $\phi(x, y)$ 超出此范围，需进行相位解包裹（Phase Unwrapping）处理。

---

## 5. 迭代FFT
上面是普通FFT提取图像参数的公式推导，这种方法有个缺点是从频域截取信号时会存在频谱泄露，导致恢复的信号与理想值存在差异，
因此需要有一种能消除频谱泄露的方法来提高相位提取精度，本算法参考下面这篇论文实现，在这我就不敲公式了。

^[Toba H, Liu Z, Udagawa S, et al. Phase analysis error reduction in the Fourier transform method using a virtual interferogram[J]. Optical Engineering, 2019, 58(8): 084103-084103.]

## 6. 核心公式速查表

| 目标参数 | 提取公式 |
| :--- | :--- |
| 背景光强 $a$ | $a = \mathcal{F}^{-1}\{ \text{LowPass}[F(u,v)] \}$ |
| +1级复信号 $c$ | $c = \mathcal{F}^{-1}\{ \text{BandPass}[F(u,v)] \}$ |
| 调制振幅 $b$ | $b = 2 \cdot \|c\|$ |
| 包裹相位 $\phi$ | $\phi = \text{atan2}(\text{Im}(c), \text{Re}(c))$ |