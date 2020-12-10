from DQN import *
from ENV import *

import datetime

action_probs = [0] * action_num
action_times = [0] * action_num


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

with tf.Session() as sess:

    start = datetime.datetime.now()
    dqn = Deep_Q_network(action_num, num_full_features)
    solution_features = tf.placeholder(tf.float32, [num_train_points,feature_size], "solution_features")
    
    #initialize_uninitialized(sess)
    #sess.run(tf.global_variables_initializer())
    for episode in range(max_episodes):

        problem = generate_problem()
        solution = construct_solution(problem)
        best_solution = copy.deepcopy(solution)
        state = generate_state()
        no_change = 0
        features = generate_solution_features(problem, solution)
        features = embedding_net(features)
        features = tf.reshape(features, [-1])
        initialize_uninitialized(sess)

        #print(sess.run(tf.report_uninitialized_variables()))

        for rollout in range(max_rollout_num):
            #确定特征shape=(68)
            local_observation = sess.run(features).tolist()

            observation = state + local_observation

            #选择动作
            action = dqn.choose_action(observation)
            next_solution, label = env_step(problem, solution, action)

            delta = solution.get_cost(problem) - next_solution.get_cost(problem)

            print(action,end="\t")
            print(delta,end="\t")  
            print(next_solution.get_cost(problem),end="\t")
            print(best_solution.get_cost(problem))

            min_delta = best_solution.get_cost(problem) - next_solution.get_cost(problem)
            reward = 0
            if delta > 0:
                solution = copy.deepcopy(next_solution)
                no_change = 0
            else:
                no_change += 1

            reward += (1 * delta)

            if min_delta > 0:
                best_solution = copy.deepcopy(next_solution)
                reward += (10 * min_delta)
            
            next_state = generate_state(state, action, reward, min_delta, delta)
            next_observation = next_state + local_observation
            
            #存储记忆
            dqn.store_transition(observation, action, reward, next_observation)
            
            #学习
            if (rollout > 1000) and (step % 10 == 0):
                dqn.learn()
            
            state = next_state

            if no_change == 10:
                break

    end = datetime.datetime.now()
    time = (start - end).total_seconds()
    print("算法总运行时间： ", time)