import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


n = 100
x = np.random.rand(100) * 6 - 3
x.sort()
y = np.sin(x) + np.random.rand(n) * 0.05
x = x.reshape(-1, 1)

reg = DecisionTreeRegressor(criterion='mse', max_depth=19)
reg.fit(x, y)
x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
y_hat = reg.predict(x_test)

plt.plot(x, y, 'r*')
plt.plot(x_test, y_hat, 'g-')
plt.show()

depth = [2, 4, 6, 8, 10]
clr = 'rgbmy'
reg = [DecisionTreeRegressor(criterion='mse', max_depth=depth[0]),
        DecisionTreeRegressor(criterion='mse', max_depth=depth[1]),
        DecisionTreeRegressor(criterion='mse', max_depth=depth[2]),
        DecisionTreeRegressor(criterion='mse', max_depth=depth[3]),
        DecisionTreeRegressor(criterion='mse', max_depth=depth[4])]

plt.plot(x, y, 'k^', linewidth=2, label='Actual')
x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
for i, r in enumerate(reg):
    dt = r.fit(x, y)
    y_hat = dt.predict(x_test)
    plt.plot(x_test, y_hat, '-', color=clr[i], linewidth=2, label='Depth=%d' % depth[i])
plt.legend(loc='upper left')
plt.grid()
plt.show()





