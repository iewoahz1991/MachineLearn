from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

'''数据准备'''
iris = datasets.load_iris()
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
x = iris['data'][:, :2]
y = iris['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

'''分步模型预测'''
# # print(np.mean(x_train[:, 0]), np.std(x_train[:, 0]))
# ss = StandardScaler()
# x_train = ss.fit_transform(x_train)
# # print(np.mean(x_train[:, 0]), np.std(x_train[:, 0]))
# model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
# model.fit(x_train, y_train)
# y_test_hat = model.predict(x_test)

'''Pipeline打包模型'''
model = Pipeline([('ss', StandardScaler()),
                  ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3))])
model.fit(x_train, y_train)
y_test_hat = model.predict(x_test)
print(y_test_hat)

'''保存'''
f = open('iris_tree.dot', 'w')
tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f)

'''画图'''
# n, m = 500, 500
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
# t1 = np.linspace(x1_min, x1_max, n)
# t2 = np.linspace(x2_min, x2_max, m)
# x1, x2 = np.meshgrid(t1, t2)
# x_show = np.stack((x1.flat, x2.flat), axis=1)
# cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
# y_show_hat = model.predict(x_show)
# y_show_hat = y_show_hat.reshape(x1.shape)
# plt.figure(facecolor='w')
# plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)
# # plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_dark)
# plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)
# plt.xlabel('花萼长度', fontsize=15)
# plt.ylabel('花萼宽度', fontsize=15)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
# plt.show()

'''计算准确率'''
result = y_test_hat == y_test
acc = np.mean(result)
print("准确率:%.2f%%"%(100 * acc))

'''过拟合错误率'''
depth = np.arange(1, 15)
err_list = []
for d in depth:
    model = model = Pipeline([('ss', StandardScaler()),
                  ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=d))])
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)
    result = y_test_hat == y_test
    err = 1 - np.mean(result)
    err_list.append(err)
    print(d, ' 错误率: %.2f%%' % (100 * err))

plt.figure(facecolor='w')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel(u'决策树深度', fontsize=15)
plt.ylabel(u'错误率', fontsize=15)
plt.title(u'决策树深度与过拟合', fontsize=17)
plt.grid(True)
plt.show()
