# Iter_FFT-cuda-accelerator

## description
A CUDA &amp; C++ accelerated version of Iter_FFT to extrcat 2D-fringe image phase. 
Compared to the Python version, the CUDA version can be 10 times faster.

Work Environment:
```
Graphics Card: RTX 4070 Super

Python 3.11

CUDA Version 13.2

CUDAToolkit 12.8

g++(Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

## quick start
We provide both the Python NumPy version and the CuPy version of the algorithm, and you can switch between them using 
the 'CUPY' parameter in cpnfig.py.
The differences in results between Python versions and CUDA versions("Python and CUDA result diff.png") are in the Iter_FFT_py folder. 
At the same time, the differences in the extracted envelope phase and ideal envelope phase using both the iterative FFT 
algorithm and the regular FFT algorithm are also in that folder("diff between the results  FFT and Iter_FFT and the ideal values.png").

### Python
```
cd Iter_FFT_py
pip install -r requirements.txt
python iter_fft.py
```

### CUDA
```
cd Iter_FFT_cu
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./app
```

## result
Here is a comparison of the Top-3 time-consuming modules

| module                     | numpy | cupy | CUDA  |
|----------------------------|-------|------|-------|
| zernike basis generate / s | 3.4   | 0.65 | 0.2   |
| zernike low pass / s       | 1.3   | 0.23 | 0.045 |
| phase unwrap / s           | 1.4   | 1.4  | 0.2   |

Here we notice that the phase unwrapping time is the same for both numpy and cupy because phase unwrapping is a highly serial algorithm. 
In Python, it calls the third-party library skimage.restoration.unwrap_phase, which doesn’t support GPU acceleration. 

So in the CUDA version, we optimized its C source code for CUDA, parallelizing the sorting and reliability map calculations, 
while keeping the union-find merge operations running on the CPU.

For the whole algorithm, our CUDA version is nearly 10 times faster compared to the numpy version. 
Of course, phase unwrapping is still a relatively time-consuming operation, so the next step is to consider other ways 
to parallelize phase unwrapping, or maybe use block-wise unwrapping (though that’s just an idea for now).