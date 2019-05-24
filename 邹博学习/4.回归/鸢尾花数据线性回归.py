import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

path = '8.iris.data'
'''手工读取'''
# f = open(path, 'r')
# x = []
# y = []
# for d in f:
#     d = d.strip()
#     d = d.split(',')
#     y.append(d[-1])
#     d = map(float, d[:-1])
#     a = []
#     for i in d:
#         a.append(i)
#     x.append(a)
# y[y == 'Iris-setosa'] = 0
# y[y == 'Iris-versicolor'] = 1
# y[y == 'Iris-virginica'] = 2

'''pandas读取'''
df = pd.read_csv(path, header=None)
x = df.values[:, :-1]
y = df.values[:, -1]
'''使用sklearn数据预处理-标签编码'''
le = preprocessing.LabelEncoder()
le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
print(le.classes_)
y = le.transform(y)

x = x[:, :2]
'''先归一化，然后逻辑回归'''
x = StandardScaler().fit_transform(x)
# print(x)
lr = LogisticRegression()
lr.fit(x, y)  #对预测的y做出转换
print(lr.coef_)

'''上述等价于'''
# lr = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression())])
# lr.fit(x, y.ravel())


'''画图'''
N, M = 500, 500
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  #第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  #第0列的范围
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)
x_test = np.stack((x1.flat, x2.flat), axis=1)
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
y_hat = lr.predict(x_test)
y_hat = y_hat.reshape(x1.shape)
#
plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()
plt.savefig('2.png')
plt.show()

'''训练集上的预测结果'''
# y_hat = lr.predict(x)
# y = y.reshape(-1)
# result = y_hat == y
# print(y_hat)
# print(result)
# acc = np.mean(result)
# print(acc)
# print('准确度: %.2f%%' % (100 * acc))


