from DQN import *
from ENV import *


import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
    #加载计算图
    #new_saver = tf.train.import_meta_graph('./Model/cvrp_100_model.ckpt.meta')



    solution_features = tf.placeholder(tf.float32, [num_train_points,feature_size], "solution_features")
    dqn = Deep_Q_network(action_num, num_full_features) # batch_size = 1记忆回放效果过低
    embed_features = embedding_net(solution_features)
    sess.run(tf.global_variables_initializer())

    #加载参数
    new_saver = tf.train.Saver()
    new_saver.restore(sess, tf.train.latest_checkpoint('./Model'))
    '''
    生成计算图
    #tensorboard --logdir=C:\python
    writer = tf.summary.FileWriter('.', tf.get_default_graph())
    writer.close()
    '''

    #将输出保存在文件results.txt中
    f = open("results.txt", "a+")
    '''
    #展示全部变量
    for v in tf.all_variables():
        print(v)
    '''
    #记录全部的全局最优解
    best_solutions = []
    #记录推理开始时间
    start = datetime.datetime.now()

    #Inference
    for episode in range(2000):
        problem = generate_problem()
        solution = construct_solution(problem, True)
        best_solution = copy.deepcopy(solution)
        state = generate_state()
        no_change = 0
        features = generate_solution_features(problem, solution)
        local_observation = sess.run(embed_features, feed_dict={solution_features:features}).reshape(-1,).tolist()
        observation = state + local_observation

        for rollout in tqdm(range(max_rollout_num), desc='episode{}_rollout:'.format(episode)):
            action = dqn.choose_action(observation)
            next_solution, label = env_step(problem, solution, action, no_change)
            next_solution.cost = next_solution.get_cost(problem)
            delta = solution.cost - next_solution.cost
            reward = 0
            if delta > 0:
                solution = copy.deepcopy(next_solution)
                reward += (1 * delta)
                no_change = 0
                features = generate_solution_features(problem, solution)
                local_observation = sess.run(embed_features, feed_dict={solution_features:features}).reshape(-1,).tolist()
            else:
                reward += (1 * delta)
                no_change += 1
                
            
            if no_change > 20:
                break
            
            
            min_delta = best_solution.cost - next_solution.cost
            if min_delta > 0:
                best_solution = copy.deepcopy(next_solution)
                reward += (10 * min_delta)
            next_state = generate_state(state, action, reward, min_delta, delta)
            next_observation = next_state + local_observation
            state = next_state
            observation = next_observation
        print(best_solution.cost)
        best_solutions.append(best_solution.cost)
    mean_cost = np.mean(best_solutions)
    print("cvrp_{}_cost: {}".format(num_train_points, mean_cost))
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    print("算法总推理时间:{}秒".format(time))