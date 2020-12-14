from DQN import *
from ENV import *

import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

action_probs = [0] * action_num
action_times = [0] * action_num


def get_color(curr_color):
    if curr_color == 0:
        return 'b'
    if curr_color == 1:
        return 'g'
    if curr_color == 2:
        return 'r'
    if curr_color == 3:
        return 'c'
    if curr_color == 4:
        return 'm'
    if curr_color == 5:
        return 'y'
    if curr_color == 6:
        return 'k'
    return 'r'


def paint(problem, solution):
    for i in range(len(problem.locations)):  # 坐标放大
        problem.locations[i][0] *= 1000
        problem.locations[i][1] *= 1000
    x = []
    y = []
    curr_color = 0
    for i in range(len(solution.path)):
        for j in range(len(solution.path[i])):
            x.append(problem.locations[solution.path[i][j]][0])
            y.append(problem.locations[solution.path[i][j]][1])
            # one route exist
        for index in range(1, len(x)):
            l_x = []
            l_y = []
            l_x.append(x[index-1])
            l_x.append(x[index])
            l_y.append(y[index-1])
            l_y.append(y[index])
            plt.plot(l_x, l_y, color=get_color(curr_color))
            plt.scatter(l_x, l_y, c='black')
        x = []
        y = []
        curr_color += 1
        if curr_color > 6:
            curr_color = 0
    plt.show()
    plt.pause(0.1)
    plt.close()


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def validate_solution(solution, problem):
    #capacity check
    for path_index in range(len(solution.path)):
        load = problem.capacities[0]
        route = solution.path[path_index]
        for node in range(1, len(route) - 1):
            node_index = route[node]
            load -= problem.capacities[node_index]
            if load < 0:
                print("当前解决方案不满足容量约束，问题在第{}个路径的{}节点".format(route, node_index))
                return False
    
    #node_number check
    nodes = np.zeros(num_train_points)
    for path_index in range(len(solution.path)):
        route = solution.path[path_index]
        for node in range(1, len(route) - 1):
            nodes[route[node] - 1] += 1
    if 0 in nodes:
        print("当前解决方案缺少节点")
        return False
    return True


with tf.Session() as sess:

    start = datetime.datetime.now()
    dqn = Deep_Q_network(action_num, num_full_features) # batch_size = 1记忆回放效果过低
    solution_features = tf.placeholder(tf.float32, [num_train_points,feature_size], "solution_features")
    embed_features = embedding_net(solution_features)
    initialize_uninitialized(sess)

    for episode in range(max_episodes):
        problem = generate_problem()
        solution = construct_solution(problem, True)
        best_solution = copy.deepcopy(solution)

        state = generate_state()
        no_change = 0
        features = generate_solution_features(problem, solution)
        #print(sess.run(tf.report_uninitialized_variables()))
        print("episode: {}".format(episode))

        for rollout in tqdm(range(max_rollout_num), desc='episode{}_rollout:'.format(episode)):
            #确定特征shape=(68)
            local_observation = sess.run(embed_features, feed_dict={solution_features:features}).reshape(-1,).tolist()
            observation = state + local_observation

            #选择动作
            action = dqn.choose_action(observation)
            next_solution, label = env_step(problem, solution, action, no_change)
            if not validate_solution(next_solution, problem):
                continue

            next_solution.cost = next_solution.get_cost(problem)
            delta = solution.cost - next_solution.cost
            
            
            #输出结果
            if rollout % 2000 == 0 and episode % 5 == 0:
                print("rollout num {}".format(rollout), end="\t")
                print("action:{}".format(action), end="\t")
                print("delta:{}".format(delta))
                print("solution_cost:{}".format(solution.cost))
                print("next_solution_cost:{}".format(next_solution.cost))
                print("best_solution_cost:{}".format(best_solution.cost))
            
            
            reward = 0
            if delta > 0:
                solution = copy.deepcopy(next_solution)
                reward += (1 * delta)
                no_change = 0
            else:
                reward += (1 * delta)
                no_change += 1

            min_delta = best_solution.cost - next_solution.cost
            if min_delta > 0:
                best_solution = copy.deepcopy(next_solution)
                #print("当前最优解为:{}".format(best_solution.cost))
                reward += (10 * min_delta)
            
            next_state = generate_state(state, action, reward, min_delta, delta)
            next_observation = next_state + local_observation
            
            #存储记忆
            dqn.store_transition(observation, action, reward, next_observation)
            
            #学习
            if rollout > 10:
                dqn.learn()
            state = next_state
        
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    print("算法总训练时间:{}秒".format(time))


    #inference
    f = open("results.txt", "a+")
    start = datetime.datetime.now()
    test_problem = generate_problem()
    solution = construct_solution(test_problem, True)
    features = generate_solution_features(test_problem, solution)
    best_solution = copy.deepcopy(solution)
    for rollout in tqdm(range(max_rollout_num), desc='Test'):
        local_observation = sess.run(embed_features, feed_dict={solution_features:features}).reshape(-1,).tolist()
        observation = state + local_observation
        #选择动作
        action = dqn.choose_action(observation)
        next_solution, label = env_step(test_problem, solution, action)
        if not validate_solution(next_solution, test_problem):
            continue
        next_solution.cost = next_solution.get_cost(test_problem)
        delta = solution.cost - next_solution.cost
        #输出rollout%50结果
        if rollout % 500 and episode % 10 == 0:
            print("rollout num {}".format(rollout), end="\t", file=f)
            print("action:{}".format(action), end="\t", file=f)
            print("delta:{}".format(delta), file=f)
            print("solution_cost:{}".format(solution.cost), file=f)
            print("next_solution_cost:{}".format(next_solution.cost), file=f)
            print("best_solution_cost:{}".format(best_solution.cost), file=f)
            f.flush()
        reward = 0
        if delta > 0:
            solution = copy.deepcopy(next_solution)
            reward += (1 * delta)
        else:
            reward += (1 * delta)

        min_delta = best_solution.cost - next_solution.cost
        if min_delta > 0:
            best_solution = copy.deepcopy(next_solution)
            #print("当前最优解为:{}".format(best_solution.cost))
            reward += (10 * min_delta)
        
        next_state = generate_state(state, action, reward, min_delta, delta)
        next_observation = next_state + local_observation
        
        #存储记忆
        dqn.store_transition(observation, action, reward, next_observation)
        
        #学习
        if rollout > 10:
            dqn.learn()
        state = next_state
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    print("算法总训练时间:{}秒".format(time), file=f)
    print("最优距离dist:{}".format(best_solution.cost()), file=f)
f.close()
