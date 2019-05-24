import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path = r'8.Advertising.csv'

'''直接读取'''
# f = open(path, 'r')
# x = []
# y = []
# for i, d in enumerate(f):
#     if i == 0:
#         continue
#     d = d.strip()
#     if not d:
#         continue
#     d = d.split(',')
#     a = []
#     d = map(float, d)
#     for j in d:
#         a.append(j)
#     x.append(a[1:-1])
#     y.append(a[-1])
# x = np.array(x)
# y = np.array(y)
'''pandas读取'''
data = pd.read_csv(path)
x = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

'''绘图1'''
# plt.plot(data['TV'], y, 'ro', label='TV')
# plt.plot(data['Radio'], y, 'g^', label='Radio')
# plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()
'''绘图2'''
# plt.figure(figsize=(9,12))
# plt.subplot(311)
# plt.plot(data['TV'], y, 'ro')
# plt.title('TV')
# plt.grid()
# plt.subplot(312)
# plt.plot(data['Radio'], y, 'g^')
# plt.title('Radio')
# plt.grid()
# plt.subplot(313)
# plt.plot(data['Newspaper'], y, 'b*')
# plt.title('Newspaper')
# plt.grid()
# plt.tight_layout()
# plt.show()
'''线性回归'''
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
linreg = LinearRegression()
model = linreg.fit(x_train, y_train)
print(model)
print(model.coef_)
print(model.intercept_)
y_hat = linreg.predict(x_test)
mse = np.average((y_hat - y_test)**2)
rmse = np.sqrt(mse)
print(mse, rmse)

t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()