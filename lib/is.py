# Python中对象包含的三个基本要素分别是：
# id(身份标识)、type(数据类型)和value(值)。
x=y=[4,5,6]
z=[4,5,6]
# ==是python标准操作符中的比较操作符
# 用来比较判断两个对象的value(值)是否相等
print(x==y)
print(x==z)
# is也被叫做同一性运算符
# 这个运算符比较判断的是对象间的唯一身份标识，也就是id是否相同。
print(x is y)
print(x is z)
# id() 函数返回对象的唯一标识符，标识符是一个整数。
# CPython 中 id() 函数用于获取对象的内存地址。
print(id(x))
print(id(y))
print(id(z))