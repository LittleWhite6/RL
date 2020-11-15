import time
import sys
 
for i in range(5):
    print (i)
    sys.stdout.flush()
    time.sleep(1)
    
'''
无论何时执行打印语句，输出都会写入缓冲区。当缓冲区被刷新（清除）时，我们将在屏幕上看到输出.
默认情况下，程序退出时将刷新缓冲区,但是我们也可以通过在程序中使用“ sys.stdout.flush（）”语句来手动刷新缓冲区.
'''