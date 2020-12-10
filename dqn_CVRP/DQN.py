from LocalSearch import *
import tensorflow as tf

#model-free: 状态转移概率未知

def generate_solution_features(problem, solution):
    features = np.zeros(shape=(num_train_points, feature_size), dtype=np.float32)
    # feature = [node_number, route_number, node_in_route_position, node_capacity, vehicle_load]

    for i in range(1, num_train_points + 1):
        features[i - 1][0] = i
        features[i - 1][3] = problem.capacities[i]

    for i in range(len(solution.path)):
        path_len = len(solution.path[i])
        load = problem.capacities[0]
        for j in range(1, path_len - 1):
            node = solution.path[i][j]
            load -= problem.capacities[node]
            features[node - 1][1] = i
            features[node - 1][2] = j
            features[node - 1][4] = load

    return features


def embed_seq(inputs, from_, to_, is_training, BN, initializer):
    with tf.variable_scope("embedding"):
        inputs = tf.expand_dims(inputs, 0)  #扩充维度做卷积操作
        W_embed = tf.get_variable("weights", [1, from_, to_], initializer=initializer)
        embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
        if BN:
            embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=False, name='layer_norm', reuse=None)
        return embedded_input


def multihead_attention(inputs, num_units, num_heads, dropout_rate, is_training):
	with tf.variable_scope("multihead_attention", reuse=None):
		# Linear projections
		Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
		K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
		V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
		# Split and concat
		Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
		K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
		V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
		# Multiplication
		outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]
		# Scale
		outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
		# Activation
		outputs = tf.nn.softmax(outputs)  # num_heads*[batch_size, seq_length, seq_length]
		# Dropouts
		outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
		# Weighted sum
		outputs = tf.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]
		# Restore shape
		outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [batch_size, seq_length, n_hidden]
		# Residual connection
		outputs += inputs  # [batch_size, seq_length, n_hidden]
		# Normalize
		outputs = tf.layers.batch_normalization(outputs, axis=2, training=True, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
	return outputs


def feedforward(inputs, num_units=[2048, 512], is_training=True):
	with tf.variable_scope("ffn", reuse=None):
		# Inner layer
		params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		# Readout layer
		params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
		outputs = tf.layers.conv1d(**params)
		# Residual connection
		outputs += inputs
		# Normalize
		outputs = tf.layers.batch_normalization(outputs, axis=2, training=True, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
	return outputs


def encode_seq(input_seq, input_dim, num_heads, num_neurons, is_training, dropout_rate=0.):
    with tf.variable_scope("encode"):
        input_seq = multihead_attention(input_seq, num_units=input_dim, num_heads=num_heads, dropout_rate=dropout_rate, is_training=is_training)
        input_seq = feedforward(input_seq, num_units=[num_neurons, input_dim], is_training=is_training)
    return input_seq


# 嵌入网络，将特征值嵌入到(64)的行向量中
def embedding_net(features):
    with tf.variable_scope('embedding_net'):
        x = embed_seq(inputs=features, from_=feature_size, to_=64, is_training=False, BN=True, initializer=tf.contrib.layers.xavier_initializer())
        layer_attention = encode_seq(input_seq=x, input_dim=64, num_heads=8, num_neurons=64, is_training=False, dropout_rate=0.1)
        layer_2 = tf.reduce_sum(layer_attention, axis=1)
    return layer_2


def generate_state(state=None, action=0, reward=0, delta_min=0, delta=0):
    if state:
        state = [action, reward, delta_min, delta]
    else:
        state = [action, reward, delta_min, delta]
    return state

class Deep_Q_network:
    #建立神经网络
    def _build_net(self):
        # 创建eval神经网络, 及时提升参数
        self.s = tf.placeholder(tf.float32, [1, self.n_features], name='s')    #接受observation: solution_features + action_num + reward + delta_min + delta
        self.q_target = tf.placeholder(tf.float32, [self.n_actions], name='Q_target')   #用来接受q_target的值，通过计算得到
        with tf.variable_scope('eval_net'):
            # c_name(collections_names)是在更新target_net参数时会用到
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64, \
                tf.random_normal_initializer(0. ,0.3), tf.constant_initializer(0.1) #config of layers   default:n_l1=10

            # eval_net的第一层. collections 是在更新target_net参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1)+b1)
            
            # eval_net的第二层，collections是在更新target_net参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
            
        with tf.variable_scope('loss'): #求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'): #梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
        # 创建target神经网络， 提供target Q
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    #接收下个observation
        with tf.variable_scope('target_net'):
            #c_names(collections_names)是在更新target_net参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            #target_net的第一层. collections是在更新target_net参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2


    def __init__(self, n_actions, n_features, learning_rate=learning_rate, reward_decay=discount_factor, e_greedy=EPSILON, replace_target_iter=300, memory_size=2000, batch_size=1, e_greedy_increment=None, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy # epsilon最大值
        self.replace_target_iter = replace_target_iter #更换target_net的步长
        self.memory_size = memory_size #记忆上限
        self.batch_size = batch_size #每次更新时从memory里取出多少记忆
        self.epsilon_increment = e_greedy_increment #epsilon的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max #是否开启探索模式，并逐步减少探索次数

        #记录学习次数（用于判断是否更换target_net参数）
        self.learn_step_counter = 0

        #初始化全0记忆[s,a,r,s_] (一个s=n_features)
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        #创建 [target_net, evaluate_net]
        self._build_net()

        #替换[target net]参数
        t_params = tf.get_collection('target_net_params')   #提取target_net的参数
        e_params = tf.get_collection('eval_net_params') #提取eval_net的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  #更新target_net参数

        self.sess = tf.Session()

        #输出tensorboard文件
        if output_graph:
            # $ tensorboard --logdir=logs (断开网络链接地址)
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_history = [] # 记录所有cost变化，用于最后plot


    #经验回放模块
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        #记录one transition
        transition = np.hstack((s, [a, r], s_))

        #总memory大小是固定的，如果超出总大小，旧memory就被新memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition #替换过程
        
        self.memory_counter += 1


    #按照policy选择行为
    def choose_action(self, observation):
        #统一observation的shape
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            #让eval_net神经网络生成所有action的值，并选择最大的action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(self.n_actions) #随机选择
        return action


    def learn(self):
        #检查是否替换target_net参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
        
        #从memory中随机抽取batch_size=32这么多记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        #获取q_next(target_net产生了q)和q_eval(eval_net产生的q)
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval], feed_dict={self.s_: batch_memory[:, -self.n_features:], self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_target = np.reshape(q_target, (10))

        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
        #self.cost_history(self.cost)    #记录cost误差

        #逐渐增加epsilon,降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()