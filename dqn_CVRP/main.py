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


def check_solution(solution, problem):
    #capacity check
    for path_index in range(len(solution.path)):
        load = problem.capacities[0]
        route = solution.path[path_index]
        for node in range(1, len(route) - 1):
            node_index = route[node]
            load -= problem.capacities[node_index]
            if load < 0:
                return False
    
    #node_number check
    nodes = np.zeros(num_train_points)
    for path_index in range(len(solution.path)):
        route = solution.path[path_index]
        for node in range(1, len(route) - 1):
            nodes[route[node] - 1] += 1
    if 0 in nodes:
        return False
    return True


with tf.Session() as sess:

    start = datetime.datetime.now()
    dqn = Deep_Q_network(action_num, num_full_features) # batch_size = 1记忆回放效果过低
    solution_features = tf.placeholder(tf.float32, [num_train_points,feature_size], "solution_features")
    embed_features = embedding_net(solution_features)
    initialize_uninitialized(sess)
    best_solution = None
    global_optimal = 0
    problem_for_check = None

    for episode in range(max_episodes):

        problem = generate_problem()
        solution = construct_solution(problem)
        if best_solution == None:
            best_solution = copy.deepcopy(solution)
        else:
            pass
        state = generate_state()
        no_change = 0
        features = generate_solution_features(problem, solution)
        min_cost = best_solution.cost
        #print(sess.run(tf.report_uninitialized_variables()))
        print("episode: {}".format(episode))

        for rollout in range(max_rollout_num):
            #确定特征shape=(68)
            local_observation = sess.run(embed_features, feed_dict={solution_features:features}).reshape(-1,).tolist()
            observation = state + local_observation

            #选择动作
            action = dqn.choose_action(observation)
            next_solution, label = env_step(problem, solution, action)
            
            if not check_solution(next_solution, problem):
                print("无效action:{}".format(action))

            solution.cost = solution.get_cost(problem)

            next_solution.cost = next_solution.get_cost(problem)

            delta = solution.cost - next_solution.cost

            #输出rollout%20结果
            if rollout%20 == 0:
                print("rollout num {}".format(rollout), end="\t")
                print("action:{}".format(action), end="\t")
                print("delta:{}".format(delta))
                print("curr_solution_cost:{}".format(solution.cost))
                print("next_solution_cost:{}".format(next_solution.cost))
                print("best_solution.cost:{}".format(best_solution.cost))

            min_delta = best_solution.cost - next_solution.cost
            reward = 0
            if delta > 1e-6:
                solution = copy.deepcopy(next_solution)
                no_change = 0
            else:
                no_change += 1

            reward += (1 * delta)

            if min_delta > 1e-6:
                best_solution = copy.deepcopy(next_solution)
                #best_solution.cost = best_solution.get_cost(problem)
                reward += (10 * min_delta)
            
            next_state = generate_state(state, action, reward, min_delta, delta)
            next_observation = next_state + local_observation
            
            #存储记忆
            dqn.store_transition(observation, action, reward, next_observation)
            
            #学习
            if rollout > 10:
                dqn.learn()
            
            state = next_state

            if no_change == 10:
                break

        min_cost_after_iter = best_solution.cost
        if min_cost_after_iter ==  min_cost:
            global_optimal += 1 
        if global_optimal == 50:
            problem_for_check = problem
            break
    
    vaild_solution = check_solution(best_solution, problem_for_check)
    if vaild_solution:
        print("解决方案有效")
    else:
        print("解决方案无效")
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    print("算法总运行时间： ", time)