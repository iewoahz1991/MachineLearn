import operator
from functools import reduce

def c(n, k):
    return reduce(operator.mul, range(n-k+1, n+1)) / reduce(operator.mul, range(1, k+1))
# 将序列中的元素逐个按operator的方式运算
