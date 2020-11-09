from time import *
count = 0
sum = 0
for i in range(1000):
    begin_time = time()
    for i in range(10000000):
        0.1024-0.5083
    end_time = time()
    run_time_1 = end_time-begin_time
    begin_time = time()
    for i in range(10000000):
        1024-5083
    end_time = time()
    run_time_2 = end_time-begin_time
    if run_time_1-run_time_2>0:
        count+=1
    sum+=1
    print(sum)
print("小数计算时间更长的概率为: {}".format(count/sum))