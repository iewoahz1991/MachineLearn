import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""解决中文显示问题"""
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

"""正态分布函数"""
# mu = 0
# sigma = 1
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 50)
# y = np.exp(-((x-mu)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
# plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
# plt.grid(True)
# plt.show()

"""胸型线"""
# x = np.arange(1, 0, -0.001)
# y = (-3 * x * np.log(x) + np.exp(-(40 * (x - 1 / np.e)) ** 4) / 25) / 2
# plt.figure(figsize=(5, 7))
# plt.plot(y, x, 'r-', linewidth=2)
# plt.grid(True)
# plt.show()

"""渐开线"""
# t = np.linspace(0, 50, num=1000)
# # x = t*np.sin(t) + np.cos(t)
# # y = np.sin(t) - t*np.cos(t)
# # plt.plot(x, y, 'r-', linewidth=2)
# # plt.grid()
# # plt.show()

"""柱状图"""
# x = np.linspace(0, 2*np.pi, 50, endpoint=True)
# y = np.sin(x)
# plt.bar(x, y, width=0.05)
# plt.plot(x, y, 'r--', linewidth=2)
# plt.title("正弦曲线")
# plt.xlabel('时间')
# plt.ylabel('幅值')
# plt.grid()
# plt.show()

"""直方图"""
# mu = 2
# sigma = 3
# data = mu + sigma * np.random.randn(1000)
# h = plt.hist(data, bins=10, normed=1, rwidth=0.5, color="r")
# # bins条形图的个数，normed =0统计频数，normed=1统计频率，rwidth条形图的宽度
# plt.grid()
# plt.show()

"""散点图"""
# 单点绘制
# plt.scatter(2, 2, s=1000, marker='o', c='r')
# plt.show()
# 分颜色绘制
x = [-1, -1, 1, 1]
y = [-1, 1, -1, 1]
z = [1, 2, 3, 4]
plt.scatter(x, y, c=z, alpha=0.5) # c等于标签区分颜色，alpha透明度
plt.show()





