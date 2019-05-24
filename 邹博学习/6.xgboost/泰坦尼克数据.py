import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

path = '12.Titanic.train.csv'
data = pd.read_csv(path)

'''性别标签处理'''
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

'''缺失船票价格处理'''
if sum(data['Fare'].isnull()) > 0:
    fare = np.zeros(3)
    for f in range(0, 3):
        fare[f] = data[data['Pclass'] == f+1]['Fare'].dropna().median()         # 按照船舱等级的中位数填充
    for f in range(0, 3):
        data.loc[(data['Fare'].isnull()) & (data['Pclass'] == f+1), 'Fare'] = fare[f]

# print(data.loc[(data['Fare'].isnull()) & (data['Pclass'] == 3), 'Fare'])
# print(data[(data['Fare'].isnull()) & (data['Pclass'] == 3)]['Fare'])          # 两者一样，单后者赋值时会报错

'''用均值填充年龄缺失值'''
# mean_age = data['Age'].dropna().mean()
# # data.loc[data['Age'].isnull(), 'Age'] = mean_age
# data['Age'].fillna(mean_age, inplace=True)

'''用随机森林预测缺失值'''

print("随机森林预测年龄缺失值：--开始--")
data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_exist = data_for_age[data_for_age['Age'].notnull()]
age_null = data_for_age[data_for_age['Age'].isnull()]
x = age_exist.values[:, 1:]
y = age_exist.values[:, 0]
rfr = RandomForestRegressor(n_estimators=1000)
rfr.fit(x, y)
age_hat = rfr.predict(age_null.values[:, 1:])
data.loc[data['Age'].isnull(), 'Age'] = age_hat
print("随机森林预测年龄缺失值：--结束--")

'''起始城市'''
data.loc[(data.Embarked == 'U'), 'Embarked'] = 'S'
embarked_data = pd.get_dummies(data.Embarked)        # 将多重标签表达为one-hot形式
print(embarked_data)
data = pd.concat([data, embarked_data], axis=1)
print(data.describe())


x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S']]
y = data['Survived']
x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

lr = LogisticRegression(penalty='l2')
lr.fit(x_train, y_train)
y_hat = lr.predict(x_test)
acc = y_hat == y_test
print("logistic回归正确率：%.2f%%" % (np.mean(acc) * 100))

lr = RandomForestClassifier(n_estimators=100)
lr.fit(x_train, y_train)
y_hat = lr.predict(x_test)
acc = y_hat == y_test
print("随机森林正确率：%.2f%%" % (np.mean(acc) * 100))


xgb_c = xgb.XGBClassifier()
xgb_c.fit(x_train, y_train)
y_hat = xgb_c.predict(x_test)
acc = y_hat == y_test
print("XGBClass正确率：%.2f%%" % (np.mean(acc) * 100))


data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 3, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic'}
bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
y_hat = bst.predict(data_test)
y_hat[y_hat > 0.5] = 1
y_hat[~(y_hat > 0.5)] = 0
acc = y_hat == y_test
print("xgboost正确率：%.2f%%" % (np.mean(acc) * 100))
















