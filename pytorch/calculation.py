from tensor import *


#加法：形式一
y = torch.rand(5, 3)
print(x + y)


#加法：形式二
print(torch.add(x, y))


#加法：给定一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)


#加法：原位/原地操作(in-place)
y.add_(x)
print(y)
#注意： 任意一个in-place改变张量的操作后面都固定一个_。例如x.copy_()、x.t_()将改变X


#也可以使用像标准的Numpy一样的各种索引操作
print(x[:, 1])


#改变形状: 如果想改变形状，可以使用torch.view()
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)   #-1表示由其他尺寸推断
print(x.size(), y.size(), z.size())


#如果是仅包含一个元素的tensor，可以使用.item()来获得对应的python数值
x = torch.randn(1)
print(x)
print(x.item())


#当GPU可用时，我们可以运行以下代码
#我们将使用'torch.device'来将tensor移入和移出GPU
if torch.cuda.is_availbale():
    device = torch.device("cuda")   #a CUDA device object
    y = torch.ones_like(x, device=device)   #直接在GPU上创建tensor
    x = x.to(device=)   #或者使用'.tp("cuda")'方法
    z = x + y  
    print(z)
    print(z.to("cpu", torch.double))    #'.to'也能在移动式改变type
    