import scipy.sparse
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def read_data(path):
    f = open(path)
    y = []
    row = []
    col = []
    values = []
    r = 0
    for d in f:
        d = d.strip().split()
        y.append(int(d[0]))
        d = d[1:]
        for c in d:
            key, value = c.split(":")
            row.append(r)
            col.append(int(key))
            values.append(float(value))
        r += 1
    x = scipy.sparse.csc_matrix((values, (row, col))).toarray()  # 将稀疏矩阵转化为稠密矩阵
    y = np.array(y)
    return x, y


def show_accuracy(a, b, tip):
    acc = a == b
    print(tip + '正确率：%.2f%%' % (np.mean(acc) * 100))


x, y = read_data("12.agaricus_train.txt")
print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5, test_size=0.4)
lr = LogisticRegression(penalty='l2')
lr.fit(x_train, y_train)
y_hat = lr.predict(x_test)
show_accuracy(y_hat, y_test, "Logistic回归")

DTC = DecisionTreeClassifier(criterion='entropy', max_depth=3)
DTC.fit(x_train, y_train)
y_hat = lr.predict(x_test)
show_accuracy(y_hat, y_test, "决策树")


data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watch_list = [(data_test, 'test'), (data_train, 'train')]
param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
y_hat = bst.predict(data_test)
show_accuracy(y_hat, y_test, 'XGBoost ')
