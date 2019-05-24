import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

iris = datasets.load_iris()
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
x_prime = iris['data']
y = iris['target']
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
for i, pairs in enumerate(feature_pairs):
    x = x_prime[:, pairs]
    clf = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=4)
    clf.fit(x, y)
    N, M = 500, 500
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    y_hat = clf.predict(x)
    c = np.count_nonzero(y_hat == y)
    print('特征：', iris_feature[pairs[0]], ' + ', iris_feature[pairs[1]])
    print('预测正确数目：', c)
    print('准确率: %.2f%%' % (100 * float(c) / float(len(y))))
    print("================")

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = clf.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同
    plt.subplot(2, 3, i + 1)
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)  # 预测值
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=cm_dark)  # 样本
    plt.xlabel(iris_feature[pairs[0]], fontsize=14)
    plt.ylabel(iris_feature[pairs[1]], fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()

plt.tight_layout(2.5)
plt.subplots_adjust(top=0.92)
plt.suptitle(u'随机森林对鸢尾花数据的两特征组合的分类结果', fontsize=18)
plt.show()
