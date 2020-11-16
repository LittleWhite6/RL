import numpy as np
step_record = np.random.uniform(size=(101, 2))
#print(step_record)

num_traversed = np.zeros((101, 101))
#print(num_traversed)

customer_indices = list(range(100 + 1))
#print(customer_indices)
import numpy as np
import datetime

f=open("results.txt",'w+')

#print(type(np.random.uniform(size=(10,2))))

#print(datetime.datetime.now())

for i in range(0,10):
    print(i,file=f)