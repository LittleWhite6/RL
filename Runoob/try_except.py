# 异常处理 try/except
# 以下例子中让用户输入一个合法的整数，但是允许用户中断这个程序(使用ctrl+c或者操作系统提供的方法)。
# 用户中断的信息会引发一个KeyboardInterrupt异常
while True:
    try:
        x=int(input("请输入一个数字："))
        break
    except ValueError:
        print("您输入的不是数字，请再次尝试输入！")

'''
try语句按照如下方式工作：
    首先，执行try子句(关键字try和关键字except之间的语句)。
    如果没有异常发生，忽略except子句，try子句执行后结束。
    如果在执行tr子句的过程中发生了异常，那么try子句余下的部分将被忽略。如果异常的类型和except之后的名称相符，那么对应的except子句将被执行。
    如果一个异常没有与任何的except匹配，那么这个异常将会被传递给上层的try中。

1)一个try语句可能包含多个except子句，分别来处理不同的特定的异常。最多只有一个分支会被执行。
2)处理程序将只针对对应的try子句中的异常进行处理，而不是其他的try的处理程序中的异常。
3)一个except子句可以同时处理多个异常，这些异常将被放在一个括号里成为一个元组，例如：
    except(RuntimeError, TypeError, NameError):
        pass
4)最后一个except子句可以忽略异常的名称，它将被当作通配符使用。你可以使用这种方法打印一个错误信息，然后再次把异常抛出。
'''

import sys
try:
    f=open('python_Runoob/myflie.txt','a+')   #ab+二进制格式
    s=f.readline()
    i=int(s.strip())
except OSError as err:
    print("OS error : {0}".format(err))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
#可以有else子句，必须放在所有except之后，如果没有发生异常则执行else部分的语句
else:
    print("None Error")
#还可以在else之后加入不管有没有异常都会执行的finally子句
finally:
    print("无论是否发生异常finally都会输出")