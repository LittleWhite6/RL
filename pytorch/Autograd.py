import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)


y = x + 2   #y是计算结果，所以它有grad_in属性
print(y)


print(y.grad_fn)


z = y * y * 3
out = z.mean()
print(z, out)
#requires_grad标志如果没有指定的话，默认输入是False
#如果指定为True, 那么它将会追踪对于该张量的所有操作


a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


#反向传播: 因为out是一个标量，因此out.backword()和out.backword(torch.tensor(1.))等价
#输出导数 d(out)/dx
print(x.grad)


#雅可比向量积的例子:
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)