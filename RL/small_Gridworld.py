import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv #Gridworld环境是构建好的，直接导入???
pp=pprint.PrettyPrinter(indent=2)
env=GridworldEnv()
#First, do the policy evaluation
def policy_eval(policy,env,discount_factor=1.0,theta=0.00001):
    #Evaluate a policy given an enviroment and a full description of the enviroment's dynamics.环境信息已知
    #Args:
    #   policy:[S,A] shaped matrix representing the policy. 策略就是从状态到动作的映射，随即策略的话这里每个方向都是1/4
    #   env. OpenAi env. env.P represents the transition probabilities of the enviroment.
    #       env.P[s][a] is a (prob,next_state,reward,done) tuple
    #   theta: We stop evaluation once our value function change is less than theta for all states. 如果前后俩次的变化小于阈值，则认为已经收敛了
    #Returns:
    #   Vector of Length env.nS representing the value function. 返回一个价值函数列表
    V=np.zeros(env.nS)
    #print("env.nS is ",env.nS) 不清楚就输出看一下
    #print(V)
    #i=0
    while True:
        delta=0
        #For each state, perform a "full backup"
        for s in range(env.nS):
            v=0
            #Look at the possible next actions
            for a,action_prob in enumerate(policy[s]):
                #print a, action_prob 这里输出就是上下左右，概率都是1/4
                #For each action, look at the possible next states.
                for prob,next_state,reward,done in env.P[s][a]:
                        #print nev.P[s][a]
                        #calculate the expected value 计算该策略下的价值函数
                        v += action_prob*prob*(reward+discount_factor*V[next_state])
                        #i=i+1
            #How much our value function changed (across any states)
            #print i,delta,v,V[s]
            delta=max(delta,np.abs(v-V[s])) #整体来讲，这个delta是先变大的，后来经过不断迭代逐渐变小，理论山趋于0的时候就是收敛的时候
            #delta=np.abs(v-V[s])
            #print(v,V[s])
            V[s]=v
        #stop evaluating once our value function change is bellow a threshold
        if delta<theta:
            break
    return np.array(V)  #最终，随即策略下的价值函数如输出所示

random_policy=np.ones([env.nS,env.nA])/env.nA
v=policy_eval(random_policy,env)

print("Value Function")
print(v)
print("")
print("Reshaped Grid Value Function:")


# Second, do the policu\y improvement 接下来通过策略迭代实现策略提升
def policy_impr(env,policy_eval_fn=policy_eval,discount_factor=1.0):
    # policy Improvement Algorithm. Iteratively evaluates and improves a policy until an optimal policy is found
    #Args:
    #   env: The OpenAI enviroment.
    #   policy_eval_fn: Policy Evaluation function that takes 3 arguments: policy,env,discount_factor. 这里要用到我们上边策略评估的结果
    #   discount_factor: Lambda discount factor
    #Returns:
    #   A tuple(policy, V) 返回最优策略和最优价值函数
    #   policy is the optimal policy, a matrix of shape [S,A].
    #       where each state s contains a valid probability distribution over actions.
    #   V is the value function for the optimal policy.
    #start with a random policy
    policy = np.ones([env.nS,env.nA])/env.nA    # 随即策略的产生
    while True:
        # Evaluate the current policy
        V= policy_eval_fn(policy,env,discount_factor)
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        # For each state ...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values=np.zeros(env.nA)
            for a in range(env.nA):
                for prob,next_state,reward,done in env.P[s][a]:
                    action_values[a] += prob * (reward+discount_factor*V[next_state])
            best_a=np.argmax(action_values)
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable=False
            policy[s]=np.eye(env.nA)[best_a]    #这里的思想就是开始先选一个行动(chosen_a)，如果这个行动和我经过计算得到的这个能产生最大价值函数的best_a一致的话，就选定。如果不是，那就是不稳定，再继续循环寻找，直到找到后break
            # print "env.nA:",env.nA
            # print np.eye(env.nA)
            # print best_a
            #print policy[s]
        if policy_stable:
            return policy,V
policy, v = policy_impr(env)
print("Policy Probability Distribution")
print(policy)
print("")
print("Reshaped Grid policy(0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy,axis=1),env.shape))
#print np.argmax(policy,axis=1)
print("")
print("Value Function:")
print(v)
print("")
print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")