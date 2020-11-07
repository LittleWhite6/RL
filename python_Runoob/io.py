s= 'Hello, Runoob'
print(str(s))
print(repr(s))
print(str(1/7))
x=10*3.25
y=200*200
#repr()函数可以转义字符串中的特殊字符
s='x的值为： ' + repr(x) +', y的值为：'+ repr(y) + '...'
print(s)
hello='hello,runoob\n'
hellos=repr(hello)
print(hellos)
print(repr((x,y,('Google','Runoob'))))