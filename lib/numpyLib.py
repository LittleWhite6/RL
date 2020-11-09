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