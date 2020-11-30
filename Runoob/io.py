import math
s = 'Hello, Runoob'
print(str(s))
print(repr(s))
print(str(1/7))
x = 10*3.25
y = 200*200
# repr()函数可以转义字符串中的特殊字符
s = 'x的值为： ' + repr(x) + ', y的值为：' + repr(y) + '...'
print(s)
hello = 'hello,runoob\n'
hellos = repr(hello)
print(hellos)
print(repr((x, y, ('Google', 'Runoob'))))


# 俩种方式输出一个平方与立方的表
for x in range(1, 11):
    print(repr(x).rjust(2), repr(x*x).rjust(3), end='')
    # 字符串对象的rjust()方法，它可以将字符串靠右，并在左边填充空格
    # 注意前一行'end'的使用
    print(repr(x*x*x).rjust(4))
# 第二种方式
for x in range(1, 11):
    print('{0:2d}{1:3d}{2:4d}'.format(x, x*x, x*x*x))


# ljust()和center()方法，不写任何东西，仅仅返回新的字符串
print(hellos.ljust(15))
print(hellos.center(10))
# zfill()方法，会在数字的左边填充0
print('12'.zfill(5))
print('3.14159265359'.zfill(5))


# str.format()基本使用方法,括号中的数字(格式化字段)用于指向传入对象在format()中的位置
print('{1}网址: "{0}!"'.format("菜鸟教程", 'www.runoob.com'))
# format()中使用了关键词参数，那么它们的值会指向使用该名字的参数
print('{name}网址: {site}'.format(name='菜鸟教程', site='www.runoob.com'))


# !a (使用ascii()), !s (使用str()), !r (使用repr())可以用于在格式化某个值之前对其进行转化：
print('常量PI的值近似为： {}'.format(math.pi))
print('常量PI的值近似为： {!r}'.format(math.pi))
# 可选项: 和格式标识符可以跟着字段名。这就允许对值进行更好的格式化。例如将pi保留到小数点后三位：
print('常量PI的值近似为{0:.3f}'.format(math.pi))
# 在 : 后传入一个整数，可以保证该域至少有这么多的宽度。用于美化表格时很有用
table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
for name, number in table.items():
    print('{0:10}==>{1:10d}'.format(name, number))
#不想分开很长的格式化字符串，可以通过变量名而非位置格式化
#传入一个字典，然后使用方括号[]来访问键值
print('Runoob: {0[Runoob]:d}; Google: {0[Google]:d}; Taobao: {0[Taobao]:d}'.format(table))
#也可以通过在table变量前使用**来实现相同的功能
print('Runoob: {Runoob:d}; Google: {Google:d}; Taobao: {Taobao:d}'.format(**table))


#读取键盘输入
#str=input("请输入：")
#print("您输入的内容是： ",str)


#读和写文件
#open(filename,mode) 返回一个file对象，mode决定了打开文件的模式，默认文件访问模式为只读(r).
#打开一个文件
f=open("python_Runoob/test.txt","a+")
f.write("测试文件读写。 \n换行成功")
str=f.read()
print(str)

#readline()会从文件中读取单独的一行。换行符为'\n'。 如果返回一个空字符串，说明已经读取到最后一行.
str=f.readline()
print(str)

#readlines()将返回该文件中包含的所有行, 如何设置可选参数sizehint,则读取指定长度的字节，并且将这些字节按行分割.
str=f.readlines()
print(str)

#迭代一个文件对象然后读取每行
for line in f:
    print(line,end='')

#f.write()将string写入到文件中，然后返回写入的字符数。
num=f.write("Python是一个非常好的语言。 \n确实")
#如果写入一些不是字符串的东西，那么将需要先进行转换：
value=('www.runoob.com',14)
del str #先前定义变量名str占用了函数名
s=str(value)
f.write(s)

#f.tell()返回文件对象当前所处的位置，它是从文件开头开始算起的字节数
f.tell()

'''
f.seek(offset,from_what)表示改变文件当前的位置
offset表示从from_what位置处往后移动offset个字符（2时表示往前移动）
form_what的值默认为0，表示文件开头; 1表示当前位置； 2表示文件的结尾
'''
f.seek(-3,2)

#f.close()关闭打开的文件并释放系统的资源，如果尝试再调用该文件则抛出异常
f.close()

#处理文件对象时，使用with关键字是非常好的方式，在结束后会帮你正确的关闭文件
with open("python_Runoob/test.txt","a+") as f:
    read_date=f.read()

#f.flush()刷新文件内部缓冲，直接把内部缓冲区的数据立刻写入文件，而不是被动的等待缓冲区写入
#f.fileno()方法返回一个整形的文件描述符(file descriptor FD 整型)，可用于底层操作系统的I/O操作
#f.truncate([size])从文件的首行首字符开始截断，截断文件为size个字符，无size表示从当前位置截断；截断之后后面的所有字符被删除，其中windows系统下的换行代表俩个字符
#f.writelines(sequence)向文件写入一个序列字符串列表，如果需要换行则要自己加入每行的换行符

f.close()

print('\t\t'+"123") # \t 表示空四个字符，也成缩进，相当于按一下Tab键
print('\n\n'+"Runoob")  # \n 表示换行

a='python'
b=a[::-1]
print(b) #nohtyp
c=a[::-2]
print(c) #nhy
#从后往前数的话，最后一个位置为-1
d=a[:-1]  #从位置0到位置-1之前的数
print(d)  #pytho
e=a[:-2]  #从位置0到位置-2之前的数
print(e)  #pyth
'''
b = a[i:j]  表示复制a[i]到a[j-1]，以生成新的list对象
 当i缺省时，默认为0，即 a[:3]相当于 a[0:3]
 当j缺省时，默认为len(alist), 即a[1:]相当于a[1:10]
 当i,j都缺省时，a[:]就相当于完整复制一份a

b = a[i:j:s]    # 表示：i,j与上面的一样，但s表示步长，缺省为1.
 所以a[i:j:1]相当于a[i:j]

 当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
 所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序。