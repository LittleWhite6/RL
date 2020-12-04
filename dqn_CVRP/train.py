from drqn import *

import datetime

action_probs = [0] * action_num
action_times = [0] * action_num

with tf.Session() as sess:

    start = datetime.datetime.now()
    
    for sample in range(max_sample_num):
        problem = generate_problem()
        solution = construct_solution(problem)
        env = env()
    
    end = datetime.datetime.now()
    time = (start - end).total_seconds()
    print("算法总运行时间： ", time)