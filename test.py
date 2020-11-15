import numpy as np
step_record = np.random.uniform(size=(101, 2))
#print(step_record)

num_traversed = np.zeros((101, 101))
#print(num_traversed)

customer_indices = list(range(100 + 1))
#print(customer_indices)

for j in range(1,101):
    print(j)