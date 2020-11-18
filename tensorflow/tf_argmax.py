import tensorflow as tf
import numpy as np
test = np.array([
[1, 2, 3],
 [2, 3, 4], 
 [5, 4, 3], 
 [8, 7, 2]])
np.argmax(test, 0)　　　#输出：array([3, 3, 1]
np.argmax(test, 1)　　　#输出：array([2, 2, 0, 0]
