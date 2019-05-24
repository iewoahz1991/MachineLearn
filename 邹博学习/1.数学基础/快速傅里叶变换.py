import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2*np.pi, 2*np.pi, 32)
y = np.sin(2*x)
f = np.fft.fft(y)
# 采样多少个点傅里叶变换也多少个点
w = np.arange(len(x))/(4*np.pi)
# 幅频特性上的频率为（n-1）fs/N,fs为采样频率
plt.plot(w, np.abs(f))
plt.show()