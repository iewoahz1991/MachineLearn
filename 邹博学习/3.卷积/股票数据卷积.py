import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

stock_max, stock_min, stock_close, stock_amount = np.loadtxt(
    '6.SH600000.txt', delimiter='\t', skiprows=2,
    usecols=(2,3,4,5), unpack=True)
N = 200
stock_close = stock_close[:N]

'''数据处理'''
n = 5
weight = np.ones(5)
weight /= weight.sum()
stock_sma = np.convolve(stock_close, weight,mode='valid')#简单移动平均

weight = np.linspace(1, 0, 5)
weight = np.exp(weight)
weight /= weight.sum()
stock_ema = np.convolve(stock_close, weight, mode='valid')#指数平均

t = np.arange(n-1, N)
poly = np.polyfit(t, stock_ema, 10) #最小二乘多项式拟合
stock_ema_hat = np.polyval(poly, t)

'''画图'''
plt.plot(np.arange(N), stock_close, 'ro-', linewidth=2, label='原始收盘价')
plt.plot(t, stock_sma, 'b-', linewidth=2, label=u'简单移动平均线')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 6))
plt.plot(np.arange(N), stock_close, 'r-', linewidth=1, label=u'原始收盘价')
plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
plt.plot(t, stock_ema_hat, 'm-', linewidth=3, label=u'指数移动平均线估计')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
