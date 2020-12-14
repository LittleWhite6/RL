from train import *

with tf.Session() as sess:
    f = open("results.txt", "a+")
    start = datetime.datetime.now()
    test_problem = generate_problem()
    solution = construct_solution(test_problem, True)
    features = generate_solution_features(test_problem, solution)
    best_solution = copy.deepcopy(solution)
    no_change = 0

    #加载模型
    saver=tf.train.import_meta_graph('C:\python\Model\cvrp_20_model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('C:\python\Model'))#这里dir2是你保存那四个文件的文件夹的绝对路径
    print("成功加载模型")

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
    print("算法总训练时间:{}秒".format(time), file=f)
    print("最优距离dist:{}".format(best_solution.cost()), file=f)
    f.close()



