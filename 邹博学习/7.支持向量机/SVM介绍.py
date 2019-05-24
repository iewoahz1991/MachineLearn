from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == '__main__':
    iris = datasets.load_iris()
    # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
    x = iris['data'][:, :2]
    y = iris['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.4)
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_train)
    acc = y_hat == y_train
    print('训练集准确率：%.2f%%' % (np.mean(acc) * 100))
    y_hat = clf.predict(x_test)
    acc = y_hat == y_test
    print('测试集准确率：%.2f%%' % (np.mean(acc) * 100))
    # 画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1, x2 = np.mgrid[x1_min:x1_max:100j, x2_min:x2_max:100j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)

    z = clf.decision_function(grid_test)
    print(z)
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)

    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
    plt.grid()
    plt.show()
