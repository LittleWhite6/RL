#numpy.random.choice(a, size=None, replace=True, p=None)
#从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
#replace:True表示可以取相同数字，False表示不可以取相同数字
#数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
import numpy as np
print(np.zeros(10)) #numpy.random.zero()生成全为0的列表
np.random.seed(1)   #随机数种子
locations=np.random.uniform(size=(10,2))
print(locations[0])  #生成随机的均匀分布列表，size表示形状    e.g. size=2表示生成一个内嵌10个元素个数为2的list的list,即10行2列的二维数组
locations[0][0]=0.05
locations[0][1]=0.05
print(locations[0])

list=[[1,2,3],[3,5,8]]
sum=np.sum(list)
print(np.sum(list))
print(np.asarray(list)/sum)

print(np.random.randint(9,size=(2,3,4)))
'''np.random.randint(low, high=None, size=None, dtype=int)
low表示下界，high表示上界(默认为None,即取值区间为(0,low] )，离散均匀分布下随机取值，区间为[low,high),数指类型为int,size表示输出的形状(int或int元组)'''


EPSILON = 1e-6
adjusted_distances = [0.1,0.2]
#,0.3,0.4,0.5,0.6,0.7,0.8,0.25
adjusted_probabilities = np.asarray([1.0 / max(d, EPSILON) for d in adjusted_distances])
print(adjusted_probabilities)
adjusted_probabilities /= np.sum(adjusted_probabilities)
print(adjusted_probabilities)


#python正无穷float("inf"), 负无穷float("-inf"), 0*float("inf")=nan
print(float("inf")+1)
print(float("inf")*0)


'''np.mean(a,axis,dtype,out,keepdims)
求均值，参数axis不设置值，对m*n个数求均值，返回一个实数
axis=0,压缩行，对各列求均值，返回1*n矩阵
axis=1,压缩列，对各行求均值，返回m*1矩阵'''
a=np.array([[1,2],[3,4]])
print(np.mean(a))
print(np.mean(a,axis=0))
print(np.mean(a,axis=1))

np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
#choice(a, size=None, replace=True, p=None)： a表示需要随机选择的列表，size表示随机选择的数据的个数，replace表示数据是否有放回，p表示每个数据被选中的概率，要与a的个数相同


'''numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值。
当一组中同时出现几个最大值时，返回第一个最大值的索引值。
axis的取值为0和1，对应剥掉的中括号，将里面的内容直接按逗号分隔：
0 —— 外层(按列找最大值索引)
1 —— 内层(按行找最大值索引)'''
one_dim_array = np.array([1, 4, 5, 3, 7, 2, 6])
print(np.argmax(one_dim_array))
two_dim_array = np.array([[1, 3, 5], [0, 4, 3]])
max_index_axis0 = np.argmax(two_dim_array, axis = 0)
max_index_axis1 = np.argmax(two_dim_array, axis = 1)
print(max_index_axis0)
print(max_index_axis1)


'''np.arange([start, ]stop, [step, ]dtype=None)
作用:   arange函数用于创建等差数组'''
nd1 = np.arange(5)      #array([0, 1, 2, 3, 4])
nd2 = np.arange(1,5)    #array([1, 2, 3, 4])
nd3 = np.arange(1,5,2)  #nd3 = np.arange(1,5,2)
print(nd1)
print(nd2)
print(nd3)